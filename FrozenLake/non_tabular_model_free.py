import numpy as np

class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)
    q = np.zeros(env.n_actions)
    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)

        done = False
        while not done:
            action = epsilon_greedy_policy(env, epsilon[i], q)
            next_state_features, reward, done = env.step(action)
            delta = reward - q[action]

            q = next_state_features.dot(theta)

            delta = delta + (gamma * np.max(q))
            theta = theta + (eta[i] * delta * features[action])
            features = next_state_features

    return theta


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)
        action = epsilon_greedy_policy(env, epsilon[i], q)
        done = False
        # TODO: Probably not correct solution. Should be fixed.
        while not done:
            next_state_features, reward, done = env.step(action)
            next_action = epsilon_greedy_policy(env, epsilon[i], q)

            current_q = q
            q = next_state_features.dot(theta)

            delta = reward + (gamma * q[next_action] - current_q[action])
            theta = theta + (eta[i] * delta * features[action])

            features = next_state_features
            action = next_action

    return theta


def epsilon_greedy_policy(env, epsilon, q):
    if np.random.uniform(0, 1) < 1 - epsilon:
        # Greedily choose best action from Q
        # Need to break ties here...
        ties = np.all(q == q[0])
        if ties:
            # All equal to zero
            action = np.random.randint(0, env.n_actions)
        else:
            action = np.argmax(q)
    else:
        action = np.random.randint(0, env.n_actions)

    return action