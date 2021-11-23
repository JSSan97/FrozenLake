import numpy as np


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    # Initialise the value function
    values = np.zeros(env.n_states, float)

    # Keep track of iterations
    current_iterations = 0

    while current_iterations < max_iterations:
        # Keep track of the update done in value function
        delta = 0
        # For every state, look ahead one step at each possible action and next state
        for s in range(env.n_states):
            v = 0
            # We would usually have a loop of all actions and probabilities but in grid world
            # there is only 1 action with 100% probability leading to a state (we are using deterministic policy)
            env.state = s
            next_state, reward, done = env.step(policy[s])
            v += env.p(next_state, s, policy[s]) * (reward + (gamma * values[next_state]))

            delta = max(delta, np.abs(v - values[s]))
            values[s] = v

        if delta < theta:
            break

        current_iterations += 1

    return values


def policy_improvement(env, policy, policy_eval, gamma):
    improved_policy = policy

    V = policy_eval

    for s in range(env.n_states):
        # Find the best action by one-step lookahead
        action_values = np.zeros(env.n_actions)

        for action in range(env.n_actions):
            env.state = s
            next_state, reward, done = env.step(action)
            action_values[action] += env.p(next_state, s, action) * (reward + (gamma * V[next_state]))

        best_action = np.argmax(action_values)
        improved_policy[s] = best_action

    return improved_policy


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    current_iterations = 0
    value = policy_evaluation(env, policy, gamma, theta, max_iterations)
    while current_iterations < max_iterations:
        policy = policy_improvement(env, policy, value, gamma)
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        current_iterations += 1

    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states, dtype=float)
    else:
        value = np.array(value, dtype=float)

    current_iterations = 0
    policy = np.zeros(env.n_states, dtype=int)

    def get_action_values(state, V):
        # Get best action
        action_values = np.zeros(env.n_actions)
        for action in range(env.n_actions):
            env.state = state
            next_state, reward, done = env.step(action)
            action_values[action] += env.p(next_state, state, action) * (reward + (gamma * V[next_state]))

        return action_values

    while current_iterations < max_iterations:
        delta = 0

        for s in range(env.n_states):
            action_values = get_action_values(s, value)
            best_action_value = np.max(action_values)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - value[s]))
            # Update the value function
            value[s] = best_action_value

        if delta < theta:
            break
        current_iterations += 1

    for s in range(env.n_states):
        action_values = get_action_values(s, value)
        best_action = np.argmax(action_values)
        policy[s] = best_action

    return policy, value
