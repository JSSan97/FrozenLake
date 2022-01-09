import numpy as np

class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        # # REPORT SECTION for big lake
        p = np.array(p)
        # p_norm = p / sum(p) # normalised for ValueError: probabilities do not sum to 1
        # next_state = self.random_state.choice(self.n_states, p=p_norm)
        if(state == 56):
            print(action)
            print(p)

        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)

        return next_state, reward
