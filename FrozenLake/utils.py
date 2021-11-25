import math
import numpy as np
import contextlib2 as contextlib


@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)

def load_npy_file():
    np.set_printoptions(threshold=np.inf)
    data = np.load('p.npy')
    return data

def get_grid_position_from_state(state, rows, columns):
    x = state % columns
    y = math.floor(state / rows)
    return x, y

def play(env):
    actions = ['w', 'a', 's', 'd']
    state = env.reset()
    env.render()

    done = False
    while not done:
        c = input('\nMove: : ')
        if c not in actions:
            raise Exception('Invalid action.')

        state, r, done = env.step(actions.index(c))
        env.render()
        print('Reward: {}.'.format(r))