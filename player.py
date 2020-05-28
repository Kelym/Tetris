from cmaes import CMAES
from features import simple_featurizer, dellacherie_featurizer, bcts_featurizer
import numpy as np
from envs.tetris import TetrisEnv
from tqdm import tqdm

N = 20

player = CMAES(bcts_featurizer)
player.load('CMAES.csv')

lines = []
env = TetrisEnv()
for i in range(N):
    done = False
    env.reset()
    while not done and env.state.cleared < 100000:
        _, reward, done, info = env.step(player.act(env.state))
    lines.append(env.state.cleared)
    print(env.state.cleared, 'moving avg', np.average(lines))

print('{} games played'.format(N))
print('Avg lines cleared', float(sum(lines)) / N)
print('Max lines cleared', float(max(lines)))
print('Lines cleared per game', lines)
