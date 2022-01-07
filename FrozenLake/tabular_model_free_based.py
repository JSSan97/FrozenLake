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

    expected = np.array([0.1892969,  0.2117458,  0.2373768,  0.26608964, 0.29824139, 0.33403301,
                0.3740742,  0.41801823, 0.20261381, 0.22729957, 0.25569216, 0.28781496,
                0.33162148, 0.37289195, 0.41890255, 0.4694183,  0.2034336,  0.22727901,
                0.25027192, 0.,         0.35567427, 0.40792664, 0.47031099, 0.52855937,
                0.22616214, 0.25460239, 0.28757367, 0.32518318, 0.38283817, 0.,
                0.51918496, 0.59522267, 0.20177241, 0.22199537, 0.24439998, 0.,
                0.44146411, 0.50665722, 0.58133736, 0.67057767, 0.17589801, 0.,
                0.,         0.41711579, 0.49147209, 0.56715201, 0.,         0.75557606,
                0.1762132,  0.,         0.3005929,  0.35401224, 0.,         0.6542872,
                0.,         0.86905418, 0.202151,   0.22713442, 0.26123183, 0.,
                0.65629287, 0.77060214, 0.86940005, 1.,         0.        ])

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

        if np.array_equal(q.max(axis=1), expected):
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

    expected = np.array([0.1892969,  0.2117458,  0.2373768,  0.26608964, 0.29824139, 0.33403301,
                0.3740742,  0.41801823, 0.20261381, 0.22729957, 0.25569216, 0.28781496,
                0.33162148, 0.37289195, 0.41890255, 0.4694183,  0.2034336,  0.22727901,
                0.25027192, 0.,         0.35567427, 0.40792664, 0.47031099, 0.52855937,
                0.22616214, 0.25460239, 0.28757367, 0.32518318, 0.38283817, 0.,
                0.51918496, 0.59522267, 0.20177241, 0.22199537, 0.24439998, 0.,
                0.44146411, 0.50665722, 0.58133736, 0.67057767, 0.17589801, 0.,
                0.,         0.41711579, 0.49147209, 0.56715201, 0.,         0.75557606,
                0.1762132,  0.,         0.3005929,  0.35401224, 0.,         0.6542872,
                0.,         0.86905418, 0.202151,   0.22713442, 0.26123183, 0.,
                0.65629287, 0.77060214, 0.86940005, 1.,         0.        ])

    for i in range(max_episodes):
        s = env.reset()
        done = False
        while not done:
            action = epsilon_greedy_policy(env, s, epsilon[i], q)
            next_state, reward, done = env.step(action)
            q[s][action] = q[s][action] + (eta[i] * (reward + (gamma*np.max(q[next_state])) - q[s][action]))
            s = next_state

        if np.array_equal(q.max(axis=1), expected):
            print("sarsa episode optimal {}".format(i))

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value
