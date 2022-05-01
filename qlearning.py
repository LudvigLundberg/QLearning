import enum
import random
from random import randrange
import numpy as np

class QLearner:
    def __init__(self, actions, learning_rate = 0.1, discount_factor = 0.9, explore_rate = 0.01, objectives = 1, weight_function = lambda x : x[0]):
        self.actions = actions
        self.Q = dict()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.explore_rate = explore_rate
        self._explore = True
        self._action_types = np.array([False, True], dtype=bool)
        self._objectives = objectives
        self._weight_function = weight_function

    def act(self, state):
        action = self.policy(state)
        self.previous_state = state
        self.previous_action = action
        return self.actions[action]

    def start_episode(self, initial_state):
        self.previous_state = initial_state
        self.previous_action = None
        if initial_state not in self.Q:
            self.Q[initial_state] = [np.array([0]*self._objectives)] * len(self.actions)

    def reward(self, reward, state):
        previous_value = self.Q[self.previous_state][self.previous_action] 
        temporal_difference = reward + self.discount_factor * self._max_estimated_future_value(state) - previous_value
        self.Q[self.previous_state][self.previous_action] = previous_value + self.learning_rate * temporal_difference

    def end_episode(self): 
        self.previous_state = None

    def _max_estimated_future_value(self, state):
        if state in self.Q:
            action_values = self.Q[state]
            values = []
            for i in range(self._objectives):
                max = action_values[0][i]
                for j,_ in enumerate(self.actions):
                    max = action_values[j][i] if action_values[j][i] > max else max
                values.append(max)
            return np.array(values)
        else:
            self.Q[state] = [np.array([0]*self._objectives)] * len(self.actions)
            return 0

    def policy(self, state):
        possible_actions = self.Q[state]
        policy_explore = np.random.choice(self._action_types, p=[1-self.explore_rate, self.explore_rate])

        if self._explore and policy_explore:
            return randrange(len(possible_actions))
        else:
            return possible_actions.index(max(possible_actions, key = self._weight_function))

    def stop_exploration(self):
        self._explore = False

    