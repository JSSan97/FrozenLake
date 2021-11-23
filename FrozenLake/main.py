from utils import play, load_npy_file
from FrozenLake import FrozenLake

# From example given in task
lake = [
    ['&', '.', '.', '.'],
    ['.', '#', '.', '#'],
    ['.', '.', '.', '#'],
    ['#', '.', '.', '$']]
slip = 0.1
max_steps = 100

frozen_lake = FrozenLake(lake, slip, max_steps)

play(frozen_lake)


