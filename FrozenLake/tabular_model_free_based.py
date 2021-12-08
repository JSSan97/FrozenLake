import numpy as np

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

    for i in range(max_episodes):
        s = env.reset()
        # TODO:
        action = epsilon_greedy_policy(env, s, epsilon[i], q)
        # While s is not terminal
        while s != env.absorbing_state:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy_policy(env, next_state, epsilon[i], q)
            q[s][action] = q[s][action] + (eta[i] * (reward + (gamma * q[next_state][next_action]) - q[s][action]))
            s = next_state
            action = next_action

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value

def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        # TODO:
        # While s is not terminal
        while s != env.absorbing_state:
            action = epsilon_greedy_policy(env, s, epsilon[i], q)
            next_state, reward, done = env.step(action)
            q[s][action] = q[s][action] + (eta[i] * (reward + (gamma*np.max(q[next_state])) - q[s][action]))
            s = next_state

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value
