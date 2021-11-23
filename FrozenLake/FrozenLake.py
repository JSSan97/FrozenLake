import numpy as np
from Environment import Environment
from utils import _printoptions, get_grid_position_from_state


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
         lake =  [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        self.slip = slip

        n_states = self.lake.size + 1
        n_actions = 4

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        self.absorbing_state = n_states - 1

        # TODO:
        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed)

        self.lake_rows, self.lake_columns = self.lake.shape

        # 0 = Up, 1 = Left, 2 = Down, 3 = Right
        self.action_state_difference = {
            0: {'x': 0, 'y': -1},
            1: {'x': -1, 'y': 0},
            2: {'x': 0, 'y': 1},
            3: {'x': 1, 'y': 0}
        }

    def step(self, action):
        state, reward, done = Environment.step(self, action)
        done = (state == self.absorbing_state) or done

        return state, reward, done

    def p(self, next_state, state, action):
        p = 0
        current_position_x, current_position_y = get_grid_position_from_state(state, self.lake_rows, self.lake_columns)
        next_state_position_x, next_state_position_y = get_grid_position_from_state(next_state, self.lake_rows,
                                                                                    self.lake_columns)

        ## Every action taken at the absorbing state leads to the absorbing state
        if state == self.absorbing_state:
            if next_state == self.absorbing_state:
                return 1
            else:
                return 0

        ## Upon taking an action at the goal or in a hole, the agent moves into the absorbing state.
        if self.lake.flat[state] == '$' or self.lake.flat[state] == '#':
            if next_state == self.absorbing_state:
                return 1
            else:
                return 0

        # ## Check if walls nearby.
        # walls = 0
        # if current_position_x == 0:
        #     walls += 1
        # if current_position_x == self.lake_columns - 1:
        #     walls += 1
        # if current_position_y == 0:
        #     walls += 1
        # if current_position_y == self.lake_rows - 1:
        #     walls += 1

        # When you intend to hit a wall
        after_moving_position_x = current_position_x + self.action_state_difference.get(action).get('x')
        after_moving_position_y = current_position_y + self.action_state_difference.get(action).get('y')
        if (after_moving_position_x < 0 or after_moving_position_y < 0
                or after_moving_position_y >= self.lake_rows or after_moving_position_x >= self.lake_columns):
            if state == next_state:
                p = (1 - self.slip)

        if next_state_position_x == after_moving_position_x and \
                next_state_position_y == after_moving_position_y:
            p = (1 - self.slip)


        ## The environment has a chance of ignoring the desired direction and the agent slips (move a random direction)
        ## Check all directions to see if the next state is adjacent to the current state, regardless of action
        for a in range(self.n_actions):
            if next_state != self.absorbing_state:
                after_moving_position_x = current_position_x + self.action_state_difference.get(a).get('x')
                after_moving_position_y = current_position_y + self.action_state_difference.get(a).get('y')

                if (after_moving_position_x < 0 or after_moving_position_y < 0 or
                        after_moving_position_y >= self.lake_rows or after_moving_position_x >= self.lake_columns):
                    if state == next_state:
                        ## If I can't go to the state because I hit a wall, the slip chance goes for the state I am on.
                        p += (self.slip / (self.n_actions))

                # If I am able to go to the state, no walls are hit, add the 'additional slip chance
                if next_state_position_x == after_moving_position_x and \
                        next_state_position_y == after_moving_position_y:

                    p += (self.slip / (self.n_actions))

        return p

    def r(self, next_state, state, action):
        r = 0

        # The agent receives reward 1 upon taking an action at the goal.
        if state != self.absorbing_state:
            if self.lake.flat[state] == '$':
                r = 1

        return r

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

                print(lake.reshape(self.lake.shape))
        else:
            actions = ['^', '<', '_', '>']
            print('Lake:')
            print(self.lake)
            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))
