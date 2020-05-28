import numpy as np
from envs.tetris import TetrisEnv
import scipy.linalg
import time
from tqdm import tqdm
import os
from multiprocessing import Pool

class Agent:
    def __init__(self, featurizer):
        self.env = TetrisEnv()
        self.env.reset()
        self.featurizer = featurizer
        self.N = len(featurizer(self.env.state, self.env.get_actions()[0]))

    def act(self, tetris_state):
        raise NotImplementedError

def sample_tetris(sample_params, featurizer, sample_len):
    def act(param, env):
        actions = env.get_actions()
        values = [param @ featurizer(env.state, a) for a in actions]
        return actions[np.argmax(values)]

    env = TetrisEnv()
    sample_pieces = np.random.randint(env.n_pieces, size=sample_len)
    m = len(sample_params)
    r, h = sample_len
    sample_scores = np.zeros((m, r))
    sample_lens = np.zeros((m, r))
    sample_clears = np.zeros((m, r))
    for i in range(m):#tqdm(range(m)):
        param = sample_params[i]
        for j in range(r):
            env.reset()
            rollout_reward = 0
            rollout_len = 0
            for _p in sample_pieces[j]:
                _, reward, done, _ = env.step(act(param, env))
                rollout_reward += reward
                rollout_len += 1
                if done:
                    break
            sample_scores[i][j] += rollout_reward
            sample_lens[i][j] += rollout_len
            sample_clears[i][j] += env.state.cleared
    return sample_scores, sample_clears, sample_lens


def eval_tetris(sample_params, featurizer, num_processes=16, sample_len=(6, 400000)):
    sample_scores, sample_clears, sample_lens = [], [], []
    pool = Pool(processes=num_processes)
    results = [pool.apply_async(sample_tetris, args=(sample_params, featurizer, sample_len)) for _ in range(num_processes)]
    pool.close()
    pool.join()
    outputs = [p.get() for p in results]
    sample_scores, sample_clears, sample_lens = zip(*outputs)
    m = len(sample_params)
    sample_scores = np.array(sample_scores).swapaxes(0,1).reshape(m,-1)
    sample_lens = np.array(sample_lens).swapaxes(0,1).reshape(m,-1)
    sample_clears = np.array(sample_clears).swapaxes(0,1).reshape(m,-1)
    sample_scores = np.average(sample_scores, axis=1)
    sample_clears = np.average(sample_clears,axis=1)
    sample_lens = np.average(sample_lens,axis=1)
    return sample_scores, {'sample_clears': sample_clears, 'sample_lens':sample_lens}

if __name__ == '__main__':
    import cma
    from features import simple_featurizer, dellacherie_featurizer, bcts_featurizer
    player = Agent(bcts_featurizer)
    es = cma.CMAEvolutionStrategy(player.N * [0], 0.5)
    best_so_far = 0
    while not es.stop():
        solutions = es.ask()
        f_vals = eval_tetris(solutions, player.featurizer)
        info = f_vals[1]
        es.tell(solutions, -f_vals[0])
        print('sample scores {} \t sample clears {}'.format(np.average(f_vals[0]), np.average(info['sample_clears'])))
        print('best scores {} \t best clears {}'.format(np.max(f_vals[0]), np.max(info['sample_clears'])))
        if np.max(info['sample_clears']) > best_so_far:
            best_so_far = np.max(info['sample_clears'])
            print('New best cleared {} lines, params: {}'.format(best_so_far, solutions[np.argmax(info['sample_clears'])]))
        es.logger.add()
        es.disp()
    es.result_pretty()
