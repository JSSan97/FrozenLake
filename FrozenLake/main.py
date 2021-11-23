from utils import play, load_npy_file
from FrozenLake import FrozenLake

################ Main function ################
def main():
    seed = 0

    # Small lake
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    env.play()


main()
