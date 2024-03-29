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
            for s_1 in range(env.n_states):
                probability = env.p(s_1, s, policy[s])
                reward = env.r(s_1, s, policy[s])
                v += probability * (reward + (gamma * values[s_1]))

            delta = max(delta, np.abs(v - values[s]))
            values[s] = v

        if delta < theta:
            break

        current_iterations += 1

    return values


def policy_improvement(env, policy, value, gamma=1):
    improved_policy = policy

    for s in range(env.n_states):
        action_values = np.zeros(env.n_actions)

        for action in range(env.n_actions):
            for next_state in range(env.n_states):
                probability = env.p(next_state, s, action)
                reward = env.r(next_state, s, action)
                action_values[action] += probability * (reward + (gamma * value[next_state]))

            best_action = np.argmax(action_values)
            improved_policy[s] = best_action

    return improved_policy

def policy_iteration(env, gamma, theta, max_iterations):
    policy = np.zeros(env.n_states, dtype=int)

    current_iterations = 0
    value = policy_evaluation(env, policy, gamma, theta, max_iterations)
    while current_iterations < max_iterations:
        policy = policy_improvement(env, policy, value, gamma)
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        current_iterations += 1

    return policy, value

def value_iteration(env, gamma, theta, max_iterations):
    policy = np.zeros(env.n_states, dtype=int)
    value = np.zeros(env.n_states, dtype=float)
    current_iterations = 0

    def get_action_values(state, V):
        # Get best action
        action_values = np.zeros(env.n_actions)
        for action in range(env.n_actions):
            for next_state in range(env.n_states):
                env.state = state
                reward = env.r(next_state, state, action)
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
