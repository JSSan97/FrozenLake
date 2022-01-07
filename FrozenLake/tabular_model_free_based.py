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

    # small lake
    # expected = np.array([2, 3, 2, 1, 2, 0, 2, 0, 3, 2, 2, 0, 0, 3, 3, 0, 0])

    # big lake
    expected = np.array([3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 2, 2, 0, 3, 3, 3, 2, 3, 3, 3, 3, 2, 0, 3, 2,
                         0, 0, 0, 0, 3, 3, 3, 2, 0, 0, 0, 3, 3, 2, 0, 2, 2, 0, 3, 0, 0, 2, 0, 2, 3, 3, 0, 0, 3, 3, 3, 0, 0])

    optimal_val = policy_evaluation(env, expected, gamma, 0.001, 100)

    printed = False
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

        val = policy_evaluation(env, q.argmax(axis=1), gamma, 0.001, 100)
        if np.array_equal(val, optimal_val) & np.array_equal(q.argmax(axis=1), expected) & (not printed):
            print("val: {}".format(val))
            print("optimal_val: {}".format(optimal_val))
            print("policy: {}".format(q.argmax(axis=1)))
            print("sarsa episode optimal {}".format(i))
            printed = True

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value

def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    # small lake
    # expected = np.array([2, 3, 2, 1, 2, 0, 2, 0, 3, 2, 2, 0, 0, 3, 3, 0, 0])

    # big lake
    expected = np.array([3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 2, 2, 0, 3, 3, 3, 2, 3, 3, 3, 3, 2, 0, 3, 2,
                         0, 0, 0, 0, 3, 3, 3, 2, 0, 0, 0, 3, 3, 2, 0, 2, 2, 0, 3, 0, 0, 2, 0, 2, 3, 3, 0, 0, 3, 3, 3, 0, 0])

    optimal_val = policy_evaluation(env, expected, gamma, 0.001, 100)

    printed = False

    for i in range(max_episodes):
        s = env.reset()
        done = False
        while not done:
            action = epsilon_greedy_policy(env, s, epsilon[i], q)
            next_state, reward, done = env.step(action)
            q[s][action] = q[s][action] + (eta[i] * (reward + (gamma*np.max(q[next_state])) - q[s][action]))
            s = next_state

        val = policy_evaluation(env, q.argmax(axis=1), gamma, 0.001, 100)
        if np.array_equal(val, optimal_val) & np.array_equal(q.argmax(axis=1), expected) & (not printed):
            print("val: {}".format(val))
            print("optimal_val: {}".format(optimal_val))
            print("policy: {}".format(q.argmax(axis=1)))
            print("q learning episode optimal {}".format(i))
            printed = True

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value
