import numpy as np
from tabular_model_based import policy_evaluation

def epsilon_greedy_policy(env, state, epsilon, q):
    if np.random.uniform(0, 1) < 1 - epsilon:
        # Greedily choose best action from Q
        # Need to break ties here...
        ties = np.all(q[state] == q[state][0])
        if ties:
            action = np.random.randint(0, env.n_actions)
        else:
            action = np.argmax(q[state])
    else:
        action = np.random.randint(0, env.n_actions)

    return action

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    # Probability of random action
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    values = []

    expected = [[0.189, 0.212, 0.237, 0.266, 0.298, 0.334, 0.374, 0.418],
                [0.203, 0.227, 0.256, 0.288, 0.332, 0.373, 0.419, 0.469],
                [0.203, 0.227, 0.25,  0.,    0.356, 0.408, 0.47,  0.529],
                [0.226, 0.255, 0.288, 0.325, 0.383, 0.,    0.519, 0.595],
                [0.202, 0.222, 0.244, 0.,    0.441, 0.507, 0.581, 0.671],
                [0.176, 0.,    0.,    0.417, 0.491, 0.567, 0.,    0.756],
                [0.176, 0.,    0.301, 0.354, 0.,    0.654, 0.,    0.869],
                [0.202, 0.227, 0.261, 0.,    0.656, 0.771, 0.869, 1.   ]]

    for i in range(max_episodes):
        s = env.reset()
        action = epsilon_greedy_policy(env, s, epsilon[i], q)
        # While s is not terminal
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy_policy(env, next_state, epsilon[i], q)
            q[s][action] = q[s][action] + (eta[i] * (reward + (gamma * q[next_state][next_action]) - q[s][action]))
            s = next_state
            action = next_action

        if np.equal(q.max(axis=1), expected):
            print("sarsa episode optimal {}".format(i))


    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value

def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    values = []

    expected = [[0.189, 0.212, 0.237, 0.266, 0.298, 0.334, 0.374, 0.418],
                [0.203, 0.227, 0.256, 0.288, 0.332, 0.373, 0.419, 0.469],
                [0.203, 0.227, 0.25,  0.,    0.356, 0.408, 0.47,  0.529],
                [0.226, 0.255, 0.288, 0.325, 0.383, 0.,    0.519, 0.595],
                [0.202, 0.222, 0.244, 0.,    0.441, 0.507, 0.581, 0.671],
                [0.176, 0.,    0.,    0.417, 0.491, 0.567, 0.,    0.756],
                [0.176, 0.,    0.301, 0.354, 0.,    0.654, 0.,    0.869],
                [0.202, 0.227, 0.261, 0.,    0.656, 0.771, 0.869, 1.   ]]

    for i in range(max_episodes):
        s = env.reset()
        done = False
        while not done:
            action = epsilon_greedy_policy(env, s, epsilon[i], q)
            next_state, reward, done = env.step(action)
            q[s][action] = q[s][action] + (eta[i] * (reward + (gamma*np.max(q[next_state])) - q[s][action]))
            s = next_state

        if np.equal(q.max(axis=1), expected):
            print("sarsa episode optimal {}".format(i))

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value
