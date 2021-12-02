from utils import play, load_npy_file
from FrozenLake import FrozenLake
from tabular_model_based import policy_iteration, value_iteration
from tabular_model_free_based import sarsa, q_learning

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

    print('')

    print('# Model-Free algorithms')
    gamma = 0.90
    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5

    print('')
    print('## Sarsa')
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)

    print('')
    print('## Q-learning')
    policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)

    print('')




    # print('')
    # print('## Numpy file')
    # print(load_npy_file())


main()
