import torch
import torch.nn as nn
import numpy as np
import os
from envs.tetris import TetrisEnv
from cmaes import Agent
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)

class DQN(Agent):
    def __init__(self, featurizer, net_size=[32,32],lr=0.001, name=''):
        super(DQN, self).__init__(featurizer)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        net_size.insert(0, self.N)
        self.net_size = net_size
        self.net = self._build_net(net_size)
        self.net.apply(init_weights)
        self.target_net = self._build_net(net_size)
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(self, self.N, featurizer)

        self.name = '{}.net={}.lr={}'.format(name, 'x'.join([str(a) for a in self.net_size]), lr)
        self.run_name = 'runs/{}-{}'.format(self.name,
            datetime.now().strftime("%m%d-%H-%M-%S"))
        self.writer = SummaryWriter(self.run_name)
        print('Save to tensorboard runs', self.run_name)

    def _build_net(self, net_size):
        layers = []
        for i in range(len(net_size) - 1):
            layers.append(nn.Linear(net_size[i], net_size[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(net_size[-1], 1))
        return nn.Sequential(*layers).to(self.device)

    def _to_tensor(self, array):
        return torch.tensor(array).float().to(self.device)

    def train(self, n_iter, train_every_iter, batch_size, cb_every_iter, discount = 0.99,
              epsilon=(128, 0.1, 0.05),
              update_target_every_iter=100, sample_per_iter=2048):

        epsilon = [float(e) for e in epsilon]
        epsilon, epsilon_decay = epsilon[0], list(epsilon[1:])

        # Train DQN
        acc_rews = []
        best_so_far = 0
        tetris_state = self.env.reset()
        count_game = 0
        total_rew = 0
        count_random = 0

        for _iter in tqdm(range(n_iter)):
            if _iter % update_target_every_iter == 0:
                self.target_net.load_state_dict(self.net.state_dict())

            # rollout
            for _is_random in np.random.sample(sample_per_iter):
                if _is_random < epsilon:
                    count_random += 1
                    actions = self.env.get_actions()
                    ac = actions[np.random.randint(len(actions))]
                else:
                    ac, v = self.act(self.env.state)
                ntetris_state, reward, done, _ = self.env.step(ac)
                total_rew += reward
                self.buffer.store(self.featurizer(tetris_state, ac), ntetris_state, reward, done)
                tetris_state = ntetris_state
                if done:
                    self.writer.add_scalar('Game/Lines', self.env.state.cleared, count_game)
                    self.writer.add_scalar('Game/Turns', self.env.state.turn, count_game)
                    self.writer.add_scalar('Game/Rews', total_rew, count_game)
                    self.writer.add_scalar('Game/Epsilon', epsilon, count_game)
                    self.writer.add_scalar('Game/ExploreTurns', count_random, count_game)
                    count_game += 1
                    count_random = 0
                    total_rew = 0
                    acc_rews.append(self.env.state.cleared)
                    tetris_state = self.env.reset()
                    break

            # train
            if self.buffer.can_sample() and _iter % train_every_iter == 0:
                self.train_step(_iter, self.buffer.sample(batch_size), discount)

            # save
            if _iter % cb_every_iter == 0:
                if np.max(acc_rews[-cb_every_iter:]) > best_so_far:
                    try:
                        os.remove('{:.2f}'.format(best_so_far))
                    except:
                        pass
                    best_so_far = np.max(acc_rews[-cb_every_iter:])
                    self.save('{:.2f}'.format(best_so_far))
                if np.average(acc_rews[-20:]) > 10000:
                    self.save('final')
                    print("Found good agent")
                    return

            if _iter and _iter % epsilon_decay[0] == 0 and epsilon > epsilon_decay[2]:
                epsilon -= epsilon_decay[1]
                if epsilon < epsilon_decay[2]: epsilon = epsilon_decay[2]
                print('Decay epsilon to', epsilon)

            #if _iter > 0 and _iter % pdb_per_iter == 0:
            #    import pdb;pdb.set_trace()
            #    pdb_per_iter *= 2

    def train_step(self, _iter, data, discount, scalar_name='Loss/train'):
        self.net.train()
        self.optimizer.zero_grad()
        state, rew, candidates, not_done = data
        output = torch.squeeze(self.net(self._to_tensor(state)), dim=1)
        label = self._to_tensor(rew + discount * np.multiply(not_done, candidates))
        loss = self.loss_fn(output, label)
        self.writer.add_scalar(scalar_name, loss, _iter)
        loss.backward()
        self.optimizer.step()

    def query_target(self, state):
        with torch.no_grad():
            return self.target_net(self._to_tensor(state)).detach().cpu().numpy()

    def query(self, tetris_state):
        self.env.state = tetris_state.copy()
        actions = self.env.get_actions()
        sa = [self.featurizer(tetris_state, action) for action in actions]
        with torch.no_grad():
            return actions, self.net(self._to_tensor(sa)).detach().cpu().numpy()

    def act(self, tetris_state):
        actions, values = self.query(tetris_state)
        return actions[np.argmax(values)], np.argmax(values)

    def save(self, fn=None):
        if fn is None:
            fn = 'torch'.format(self.net_size)
        fn = '{}-{}.pth'.format(self.name, fn)
        print('Saved model to {}'.format(fn))
        torch.save({
            'net':self.net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'buffer':self.buffer.state_dict()
            }, fn)

    def load(self, fn='torch'):
        fn = '{}-{}.pth'.format(self.name, fn)
        state_dict = torch.load(fn)
        self.net.load_state_dict(state_dict['net'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.buffer.load_state_dict(state_dict['buffer'])
        self.target_net.load_state_dict(state_dict['target_net'])

class ReplayBuffer():
    def __init__(self, agent, state_size, featurizer,
                 ready_size=2048, capacity=1024*64):
        self.agent = agent
        self.state_size = state_size
        self.featurizer = featurizer
        self.ready_size = ready_size
        self.capacity = m = capacity
        self.idx = 0
        self.num_in_buffer = 0
        self.env = TetrisEnv()
        self.env.reset()
        moves = [len(self.env.legal_moves[i]) for i in range(self.env.n_pieces)]
        self.num_moves = sum(moves)
        self.moves_per_pieces = moves
        self.state = np.zeros((m,state_size))
        self.nstate = np.zeros((m, self.num_moves, state_size))
        self.rew = np.zeros(m)
        self.done = np.zeros(m)

    def state_dict(self):
        return {
            'capacity': self.capacity,
            'idx': self.idx,
            'num_in_buffer': self.num_in_buffer,
            'state': self.state,
            'nstate': self.nstate,
            'rew': self.rew,
            'done': self.done,
        }

    def load_state_dict(self, _dict):
        self.capacity = _dict['capacity']
        self.idx = _dict['idx']
        self.num_in_buffer = _dict['num_in_buffer']
        self.state = _dict['state']
        self.nstate = _dict['nstate']
        self.rew = _dict['rew']
        self.done = _dict['done']

    def save(self, fn='Expert_buffer.pth'):
        torch.save(self.state_dict(), fn)

    def load(self, fn='Expert_buffer.pth'):
        self.load_state_dict(torch.load(fn))

    def can_sample(self):
        return self.num_in_buffer >= self.ready_size

    def store(self, state, ntetris_state, reward, done):
        i = self.idx
        self.state[i] = state
        self.rew[i] = reward
        self.done[i] = done
        counter = 0
        for _p in range(self.env.n_pieces):
            self.env.state.next_piece = _p
            for ac in self.env.get_actions():
                self.nstate[i][counter] = self.featurizer(self.env.state, ac)
                counter += 1
        self.idx = (self.idx + 1) % self.capacity
        if self.num_in_buffer < self.capacity: self.num_in_buffer += 1

    def sample(self, batch_size, chosen_idx=None):
        if chosen_idx is None:
            chosen_idx = np.random.choice(self.num_in_buffer, batch_size, replace=False)
        candidates = self.nstate[chosen_idx].reshape(-1, self.state_size)
        scores = self.agent.query_target(candidates).reshape(batch_size, self.num_moves)
        start_idx = 0
        exp_scores = np.zeros((batch_size, self.env.n_pieces))
        for i, num_move in enumerate(self.moves_per_pieces):
            exp_scores[:,i] = np.max(scores[:, start_idx:start_idx+num_move])
            start_idx += num_move
        exp_scores = np.average(exp_scores, axis=1)
        return self.state[chosen_idx], self.rew[chosen_idx], exp_scores, 1 - self.done[chosen_idx]

    def iterator(self, batch_size):
        shuffle_idx = np.arange(self.num_in_buffer)
        np.random.shuffle(shuffle_idx)
        start_idx = 0
        while start_idx + batch_size < self.num_in_buffer:
            yield self.sample(batch_size, shuffle_idx[start_idx:start_idx+batch_size])
            start_idx += batch_size

if __name__ == '__main__':
    from features import simple_featurizer, dellacherie_featurizer, bcts_featurizer, column_featurizer
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--name', type=str, default='DQN')
    parser.add_argument('--warmup_episodes', type=int, default=0)
    parser.add_argument('--epsilon', nargs=4, default=(.95, 128, .1, .05))
    parser.add_argument('--epoch', type=int, default=500000)
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()
    agent = DQN(bcts_featurizer, net_size=[32,32], lr=args.lr, name=args.name)
    agent.train(args.epoch, 1, args.batch_size, 64, epsilon=args.epsilon)
    agent.save()
