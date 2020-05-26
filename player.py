from cmaes import CMAES
from features import simple_featurizer, dellacherie_featurizer, bcts_featurizer
import numpy as np
from envs.tetris import TetrisEnv
from tqdm import tqdm

N = 40

player = CMAES(simple_featurizer)
player.load('4065.5.csv')
player.params = np.array([-0.510066, -0.35663, -0.184483, 0.760666])

lines = []
env = TetrisEnv()
for i in range(N):
    done = False
    env.reset()
    while not done and env.state.cleared < 100000:
        _, reward, done, info = env.step(player.act(env.state))
    print(env.state.cleared)
    lines.append(env.state.cleared)

print('{} games played'.format(N))
print('Avg lines cleared', float(sum(lines)) / N)
print('Max lines cleared', float(max(lines)))
print('Lines cleared per game', lines)