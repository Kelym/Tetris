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
    for i in tqdm(range(m)):
        param = sample_params[i]
        for j in range(r):
            env.reset()
            rollout_reward = 0
            rollout_len = 0
            for _p in sample_pieces[j]:
            #for _ in range(h):
                #env.state.next_piece = _p
                #env.render()
                #time.sleep(0.1)
                _, reward, done, _ = env.step(act(param, env))
                rollout_reward += reward
                rollout_len += 1
                if done:
                    break
            sample_scores[i][j] += rollout_reward
            sample_lens[i][j] += rollout_len
            sample_clears[i][j] += env.state.cleared
    return sample_scores, sample_clears, sample_lens



class CMAES(Agent):
    def __init__(self, featurizer):
        super(CMAES, self).__init__(featurizer)
        self.params = np.zeros(self.N)
        self.eval = self.eval_tetris

    def eval_tetris(self, sample_params, num_processes=16, sample_len=(6, 400000)):
        sample_scores, sample_clears, sample_lens = [], [], []
        pool = Pool(processes=num_processes)
        results = [pool.apply_async(sample_tetris, args=(sample_params, self.featurizer, sample_len)) for _ in range(num_processes)]
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

    def eval_toy(self, sample_params):
        ground_truth = np.array([0.5, 0.5])
        def f(x): return - np.linalg.norm(x - ground_truth)
        sample_scores = np.array([f(x) for x in sample_params])
        return sample_scores, None

    def train(self, callback_per_iter, m, mean, sigma):
        n = self.N
        cov = np.eye(n)
        sel = m // 2

        # Weights
        # weights = np.array([sel + 1 - i for i in range(sel)]) # linear weights
        weights = np.log(sel + 0.5) - np.log(np.arange(1, sel+1)) # superlinear weights, avoid negative
        weights = weights / np.sum(weights)
        ueff = 1. / np.sum(weights ** 2)
        print("lambda = {}, sel = {}, w = {}, ueff = {}".format(m, sel, weights, ueff))

        # Stupid learning rates .... ahhhhhhhhh I am so scared of messing things up!
        ps = np.zeros(n)
        pc = np.zeros(n)
        chiN = np.sqrt(n) * (1- 1./(4*n) + 1./(21 * (n ** 2)))
        # Hansen's Table 1, page 31 from tutorial
        cs = (ueff + 2) / (n + ueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((ueff - 1)/(n+1))-1) + cs
        cc = (4 + ueff / n) / (n + 4 + 2 * ueff / n)
        c1 = 2 / ((n+1.3) ** 2 + ueff)
        cu = min(1-c1, 2*(ueff - 2 + 1. / ueff) / ((n+2)**2 + ueff))

        best_so_far = -float('inf')
        for _iter in range(1000):
            # Sample
            z = np.random.multivariate_normal(np.zeros(n), cov, size=m)
            sample_params = mean + z * sigma
            # FOR TETRIS:
            sample_params = sample_params / np.linalg.norm(sample_params, axis=1, keepdims=True) # weight normalization comes from Boumaza 2013

            # Evaluate
            sample_scores, info = self.eval(sample_params)

            # Update
            sort_ind = np.argsort(-sample_scores)
            sel_ind = sort_ind[:sel]
            old_mean = mean.copy()
            mean = np.average(sample_params[sel_ind], axis=0, weights=weights)
            z_ = (sample_params[sel_ind] - mean) / sigma
            y = np.average(z_, axis=0, weights=weights)
            Dsquare, B = np.linalg.eigh(cov)
            invsqrtC = B @ (1./np.sqrt(Dsquare) * B.T)
            ps = (1 - cs) * ps + np.sqrt(cs * (2-cs) * ueff) * invsqrtC @ y
            hs = 1 if np.linalg.norm(ps) < ((1.4 + 2/(n+1)) * chiN * np.sqrt(1 - (1-cs) ** (2 * (_iter+2)))) else 0
            pc = (1 - cc) * pc + hs * np.sqrt(cc * (2-cc) * ueff) * y
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1 ))
            ncov = (z_ * weights[:,None]).T @ z_
            cov = (1 + c1 * (1-hs) * cc (2-cc) - c1 - cu) * cov + c1 * pc @ pc.T + cu * ncov

            #print(mean, sigma, cov)
            info['sel_ind'] = sel_ind
            if callback_per_iter(_iter, sample_params, sample_scores, info):
                print('Stopping criteria reached')
                self.save()
                break

            sample_clears = info['sample_clears']
            if sample_clears[sel_ind[0]] > best_so_far + 100:
                if best_so_far > -float('inf'):
                    os.remove('{}.csv'.format(best_so_far))
                self.params = sample_params[sel_ind[0]]
                best_so_far = sample_clears[sel_ind[0]]
                self.save('{}.csv'.format(best_so_far))

        self.params = sample_params[sel_ind[0]]

    def act(self, tetris_state):
        self.env.state = tetris_state.copy()
        actions = self.env.get_actions()
        values = [self.params @ self.featurizer(tetris_state, a) for a in actions]
        return actions[np.argmax(values)]

    def save(self, fn='CMAES.csv'):
        from numpy import asarray
        from numpy import savetxt
        print('Saved agent to {}'.format(fn))
        savetxt(fn, self.params, delimiter=',')

    def load(self, fn='CMAES.csv'):
        from numpy import loadtxt
        self.params = loadtxt(fn, delimiter=',')

def gen_cb_toy():
    import matplotlib.pyplot as plt
    plt.ion()
    fig, ax = plt.subplots()
    x, y = [],[]
    sc = ax.scatter(x,y,s=1,)
    plt.xlim(-1,20)
    plt.ylim(-1,20)
    plt.draw()
    def callback_toy(_iter, sample_params, sample_scores, info):
        print('Iter {} \t avg f() {}'.format(_iter, np.average(sample_scores)))
        x = sample_params[:,0]
        y = sample_params[:,1]
        sc.set_offsets(np.c_[x,y])
        fig.canvas.draw_idle()
        plt.pause(0.3)
        return np.average(sample_scores) > -1e-10
    return callback_toy

def callback_tetris(_iter, sample_params, sample_scores, info):
    #import pdb;pdb.set_trace()
    print('Iter {} \t Cleared {:.2f} \t Len {:.2f}'.format(_iter, np.average(info['sample_clears']), np.average(info['sample_lens'])))
    print('Best cleared {} and lasted {}'.format(np.max(info['sample_clears']), np.max(info['sample_lens'])))
    print('Best params: {}'.format(sample_params[info['sel_ind'][0]]))
    # Stopping criteria
    if info['sample_clears'][info['sel_ind'][0]] >= 10000:
        print('Found a set of good parameters!')
        return True

if __name__ == '__main__':
    from features import simple_featurizer, dellacherie_featurizer, bcts_featurizer
    player = CMAES(bcts_featurizer)

    # Toy
    #player.N = 2
    #player.eval = player.eval_toy
    #player.train(gen_cb_toy(), m=64, mean=np.random.uniform(0, 10, n), sigma=0.3)

    init_mean = np.zeros(player.N)
    m = int(4 + 3 * log(player.N))
    player.train(callback_tetris, m=m, mean=init_mean, sigma=0.5)
    print('Finished!')
