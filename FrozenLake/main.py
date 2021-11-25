from utils import play, load_npy_file
from FrozenLake import FrozenLake
from tabular_model_based import policy_iteration, value_iteration

################ Main function ################
def main():
    seed = 0

    # Small lake
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    env.reset()
    # play(env)

    print('# Model-based algorithms')
    gamma = 0.90
    theta = 0.001
    max_iterations = 100

    print('')

    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')

    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    # print('')
    # print('## Numpy file')
    # print(load_npy_file())


main()
