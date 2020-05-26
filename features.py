import numpy as np
from envs.tetris import TetrisEnv

dummy_env = TetrisEnv()

# =====================================================
# Features recommended in
# https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/
# with slight modification (since we are using Q learning instead)

def simple_featurizer(tetris_state, action):
    orient, slot = action
    # landing_height
    _height = max(
        tetris_state.top[slot+c] - dummy_env.piece_bottom[tetris_state.next_piece][orient][c]
        for c in range(dummy_env.piece_width[tetris_state.next_piece][orient])
    )
    # adjust top
    _top = tetris_state.top.copy()
    for c in range(dummy_env.piece_width[tetris_state.next_piece][orient]):
        _top[slot+c] = _height + dummy_env.piece_top[tetris_state.next_piece][orient][c]
    # adjust field
    if max(_top) < dummy_env.n_rows:
        _field = tetris_state.field.copy()
    else:
        _field = np.zeros((max(_top), dummy_env.n_cols))
        _field[:dummy_env.n_rows, :] = tetris_state.field.copy()
    turn = tetris_state.turn + 1
    for i in range(dummy_env.piece_width[tetris_state.next_piece][orient]):
        for h in range(_height + dummy_env.piece_bottom[tetris_state.next_piece][orient][i],
                       _height + dummy_env.piece_top[tetris_state.next_piece][orient][i]):
            _field[h, i+slot] = turn
    # complete and eroded
    counter_complete_lines = 0
    counter_contribution = 0
    for i in range(dummy_env.n_rows):
        if np.all(_field[i] > 0):
            counter_complete_lines += 1
            counter_contribution += np.sum(_field[i] == turn)
    # binary the field
    bi_field = np.where(_field != 0, 1, 0)
    count_holes = _top.sum() - bi_field.sum()
    return [
        _top.sum(), # aggregate height
        count_holes, # count holes
        np.abs(_top[:-1] - _top[1:]).sum(), # wall bump
        counter_complete_lines,
        #_top.max()
        ]

def agg_height(tetris_state):
    return tetris_state.top.sum()

def count_holes(tetris_state):
    bi_field = np.where(tetris_state.field != 0, 1, 0)
    return np.sum(bi_field[1:] - bi_field[:-1] == 1)

def wall_bump(tetris_state):
    top = tetris_state.top
    return np.abs(top[:-1] - top[1:]).sum()

def complete_lines(tetris_state, action):
    orient, slot = action
    turn = tetris_state.turn + 1
    _height = landing_height(tetris_state, action)
    if _height + dummy_env.piece_height[tetris_state.next_piece][orient] >= dummy_env.n_rows:
        return -100
    _field = tetris_state.field.copy()
    for i in range(dummy_env.piece_width[tetris_state.next_piece][orient]):
        for h in range(_height + dummy_env.piece_bottom[tetris_state.next_piece][orient][i],
                       _height + dummy_env.piece_top[tetris_state.next_piece][orient][i]):
            _field[h, i+slot] = turn
    counter_complete_lines = 0
    for i in range(dummy_env.n_rows):
        if np.all(_field[i] > 0):
            counter_complete_lines += 1
    return counter_complete_lines

# =====================================================
# Features recommended in (Boumaza 2013)
#

def bcts_featurizer(tetris_state, action):
    orient, slot = action
    _height = max(
        tetris_state.top[slot+c] - dummy_env.piece_bottom[tetris_state.next_piece][orient][c]
        for c in range(dummy_env.piece_width[tetris_state.next_piece][orient])
    ) # landing_height
    if _height + dummy_env.piece_height[tetris_state.next_piece][orient] >= dummy_env.n_rows:
        return [0] * 8 # dead
    # adjust field
    _field = tetris_state.field.copy()
    turn = tetris_state.turn + 1
    for i in range(dummy_env.piece_width[tetris_state.next_piece][orient]):
        for h in range(_height + dummy_env.piece_bottom[tetris_state.next_piece][orient][i],
                       _height + dummy_env.piece_top[tetris_state.next_piece][orient][i]):
            _field[h, i+slot] = turn
    # adjust top
    _top = tetris_state.top.copy()
    for c in range(dummy_env.piece_width[tetris_state.next_piece][orient]):
        _top[slot+c] = _height + dummy_env.piece_top[tetris_state.next_piece][orient][c]

    # complete and eroded
    counter_complete_lines = 0
    counter_contribution = 0
    for i in range(dummy_env.n_rows):
        if np.all(_field[i] > 0):
            counter_complete_lines += 1
            counter_contribution += np.sum(_field[i] == turn)

    # row_trans
    bi_field = np.zeros((dummy_env.n_rows + 1, dummy_env.n_cols))
    bi_field[0] = np.asarray([1 if _top[j] > 0 else 0 for j in range(dummy_env.n_cols)])
    bi_field[1:,:] = np.where(_field != 0, 1, 0)
    count_row_trans = np.sum(bi_field[1:] - bi_field[:-1] == 1) + np.sum(bi_field[:-1] - bi_field[1:] == 1)

    # col_trans
    bi_field = np.zeros((dummy_env.n_rows + 1, dummy_env.n_cols))
    bi_field[0] = np.asarray([1 if tetris_state.top[j] > 0 else 0 for j in range(dummy_env.n_cols)])
    bi_field[1:,:] = np.where(tetris_state.field != 0, 1, 0)
    count_col_trans = np.sum(bi_field[1:] - bi_field[:-1] == 1) + np.sum(bi_field[:-1] - bi_field[1:] == 1)

    # wall_cumu
    counter_wall_cumu = 0
    if _top[1] > _top[0]:
        counter_wall_cumu += _top[1] - _top[0]
    if _top[-2] > _top[-1]:
        counter_wall_cumu += _top[-2] - _top[-1]
    for i in range(dummy_env.n_cols - 2):
        if _top[i] > _top[i+1] and _top[i+2] > _top[i+1]:
            counter_wall_cumu += min(_top[i] - _top[i+1], _top[i+2] - _top[i+1])

    # binary the field
    bi_field = np.where(_field != 0, 1, 0)
    count_holes = np.sum(bi_field[1:] - bi_field[:-1] == 1)

    # row holes
    row_holes = (bi_field[1:] - bi_field[:-1] == 1)
    row_holes = np.any(row_holes, axis=1).sum()

    # hole depth
    counter_hole_depth = 0
    _first_h = (_field == 0).argmax(axis = 0)
    for i in range(dummy_env.n_cols):
        counter_hole_depth += (_field[_first_h[i]:,i] != 0).sum()

    return [count_holes,
            _height,            # landing_height
            count_row_trans,
            count_col_trans,
            counter_wall_cumu,
            counter_contribution * counter_complete_lines,
            counter_hole_depth,
            row_holes,
            ]

def dellacherie_featurizer(tetris_state, action):
    orient, slot = action
    _height = max(
        tetris_state.top[slot+c] - dummy_env.piece_bottom[tetris_state.next_piece][orient][c]
        for c in range(dummy_env.piece_width[tetris_state.next_piece][orient])
    ) # landing_height
    if _height + dummy_env.piece_height[tetris_state.next_piece][orient] >= dummy_env.n_rows:
        return [0] * 6 # dead
    # adjust field
    _field = tetris_state.field.copy()
    turn = tetris_state.turn + 1
    for i in range(dummy_env.piece_width[tetris_state.next_piece][orient]):
        for h in range(_height + dummy_env.piece_bottom[tetris_state.next_piece][orient][i],
                       _height + dummy_env.piece_top[tetris_state.next_piece][orient][i]):
            _field[h, i+slot] = turn
    # adjust top
    _top = tetris_state.top.copy()
    for c in range(dummy_env.piece_width[tetris_state.next_piece][orient]):
        _top[slot+c] = _height + dummy_env.piece_top[tetris_state.next_piece][orient][c]

    # complete and eroded
    counter_complete_lines = 0
    counter_contribution = 0
    for i in range(dummy_env.n_rows):
        if np.all(_field[i] > 0):
            counter_complete_lines += 1
            counter_contribution += np.sum(_field[i] == turn)

    # row_trans
    bi_field = np.zeros((dummy_env.n_rows + 1, dummy_env.n_cols))
    bi_field[0] = np.asarray([1 if _top[j] > 0 else 0 for j in range(dummy_env.n_cols)])
    bi_field[1:,:] = np.where(_field != 0, 1, 0)
    count_row_trans = np.sum(bi_field[1:] - bi_field[:-1] == 1) + np.sum(bi_field[:-1] - bi_field[1:] == 1)

    # col_trans
    bi_field = np.zeros((dummy_env.n_rows + 1, dummy_env.n_cols))
    bi_field[0] = np.asarray([1 if tetris_state.top[j] > 0 else 0 for j in range(dummy_env.n_cols)])
    bi_field[1:,:] = np.where(tetris_state.field != 0, 1, 0)
    count_col_trans = np.sum(bi_field[1:] - bi_field[:-1] == 1) + np.sum(bi_field[:-1] - bi_field[1:] == 1)

    # wall_cumu
    counter_wall_cumu = 0
    if _top[1] > _top[0]:
        counter_wall_cumu += _top[1] - _top[0]
    if _top[-2] > _top[-1]:
        counter_wall_cumu += _top[-2] - _top[-1]
    for i in range(dummy_env.n_cols - 2):
        if _top[i] > _top[i+1] and _top[i+2] > _top[i+1]:
            counter_wall_cumu += min(_top[i] - _top[i+1], _top[i+2] - _top[i+1])

    # binary the field
    bi_field = np.where(_field != 0, 1, 0)
    count_holes = np.sum(bi_field[1:] - bi_field[:-1] == 1)

    return [count_holes,
            _height,            # landing_height
            count_row_trans,
            count_col_trans,
            counter_wall_cumu,
            counter_contribution * counter_complete_lines]

def landing_height(tetris_state, action):
    orient, slot = action
    return max(
        tetris_state.top[slot+c] - dummy_env.piece_bottom[tetris_state.next_piece][orient][c]
        for c in range(dummy_env.piece_width[tetris_state.next_piece][orient])
    )

def eroded_pieces(tetris_state, action):
    orient, slot = action
    turn = tetris_state.turn + 1
    _height = landing_height(tetris_state, action)
    if _height + dummy_env.piece_height[tetris_state.next_piece][orient] >= dummy_env.n_rows:
        return -100 * counter_contribution
    _field = tetris_state.field.copy()
    for i in range(dummy_env.piece_width[tetris_state.next_piece][orient]):
        for h in range(_height + dummy_env.piece_bottom[tetris_state.next_piece][orient][i],
                       _height + dummy_env.piece_top[tetris_state.next_piece][orient][i]):
            _field[h, i+slot] = turn
    counter_complete_lines = 0
    counter_contribution = 0
    for i in range(dummy_env.n_rows):
        if np.all(_field[i] > 0):
            counter_complete_lines += 1
            counter_contribution += np.sum(_field[i] == turn)
    return counter_complete_lines * counter_contribution

def row_transitions(tetris_state):
    bi_field = np.zeros((dummy_env.n_rows, dummy_env.n_cols + 2))
    bi_field[:tetris_state.top[0],0] = 1
    bi_field[:tetris_state.top[-1],-1] = 1
    bi_field[:,1:-1] = np.where(tetris_state.field != 0, 1, 0)
    return np.sum(bi_field[:,1:] - bi_field[:,:-1] == 1) + np.sum(bi_field[:,:-1] - bi_field[:,1:] == 1)

def col_transitions(tetris_state):
    bi_field = np.zeros((dummy_env.n_rows + 1, dummy_env.n_cols))
    bi_field[0] = np.asarray([1 if tetris_state.top[j] > 0 else 0 for j in range(dummy_env.n_cols)])
    bi_field[1:,:] = np.where(tetris_state.field != 0, 1, 0)
    return np.sum(bi_field[1:] - bi_field[:-1] == 1) + np.sum(bi_field[:-1] - bi_field[1:] == 1)

def wall_cumu(tetris_state):
    _top = tetris_state.top
    counter = 0
    if _top[1] > _top[0]:
        counter += _top[1] - _top[0]
    if _top[-2] > _top[-1]:
        counter += _top[-2] - _top[-1]
    for i in range(dummy_env.n_cols - 2):
        if _top[i] > _top[i+1] and _top[i+2] > _top[i+1]:
            counter += min(_top[i] - _top[i+1], _top[i+2] - _top[i+1])
    return counter

def hole_depth(tetris_state):
    counter = 0
    _first_h = (tetris_state.field == 0).argmax(axis = 0)
    for i in range(dummy_env.n_cols):
        counter += (tetris_state.field[_first_h[i]:,i] != 0).sum()
    return counter

def row_hole(tetris_state):
    bi_field = np.where(tetris_state.field != 0, 1, 0)
    holes = (bi_field[1:] - bi_field[:-1] == 1)
    return np.any(holes, axis=1).sum()

'''
def simple_featurizer(tetris_state, action):
    return np.array([agg_height(tetris_state),
                     complete_lines(tetris_state, action),
                     count_holes(tetris_state),
                     wall_bump(tetris_state)])

def dellacherie_featurizer(tetris_state, action):
    return np.array([count_holes(tetris_state),
                     landing_height(tetris_state, action),
                     row_transitions(tetris_state),
                     col_transitions(tetris_state),
                     wall_cumu(tetris_state),
                     eroded_pieces(tetris_state, action)])

def bcts_featurizer(tetris_state, action):
    return np.concatenate(
            (dellacherie_featurizer(tetris_state, action),
             [hole_depth(tetris_state),row_hole(tetris_state)]))
'''

def print_state(state, ac):
    print('agg height\t', agg_height(state))
    print('count holes\t', count_holes(state))
    print('wall bump\t', wall_bump(state))
    print('row trans\t', row_transitions(state))
    print('col trans\t', col_transitions(state))
    print('wall cumu\t', wall_cumu(state))
    print('hole depth\t', hole_depth(state))
    print('row hole\t', row_hole(state))
    print(simple_featurizer(state,ac))
    print(dellacherie_featurizer(state, ac))
    print(bcts_featurizer(state, ac))

if __name__ == '__main__':
    env = TetrisEnv()
    lstate = env.reset()
    for _ in range(20):
        actions = env.get_actions()
        action = actions[np.random.randint(len(actions))]
        nstate, reward, done, _ = env.step(action)
        if done:
            env.state = lstate.copy()
            env.render(show_action=True)
            print_state(lstate, action)
            break
        lstate = nstate
