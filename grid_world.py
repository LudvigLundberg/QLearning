from abc import ABC, abstractmethod
import numpy as np

class GridWorld(ABC):
    def __init__(self, height, width):
        self.actions = ['up','down','right','left']
        self._height = height
        self._width = width

    @abstractmethod
    def current_state(self):
        pass

    @abstractmethod
    def act(self, action):
        pass

    def out_of_bounds(self, action):
        match action:
            case 'up':
                return self.actor_pos < self._width
            case 'down':
                return self.actor_pos >= (self._height*self._width) - self._width
            case 'left':
                return self.actor_pos % self._width == 0
            case 'right':
                return self.actor_pos % self._width == self._width - 1

class Default(GridWorld):
    def __init__(self, board, rewards, start, goal, height, width):
        self._height = height
        self._width = width
        self.state = board
        self.rewards = rewards
        self.actor_pos = start
        self.goal = goal

    def current_state(self):
        return tuple(self.state)

    def act(self, action):
        if self.out_of_bounds(action):
            return np.array([-10,-10]), False
        
        self.state[self.actor_pos] = 0
        match action:
            case 'up':
                self.actor_pos = self.actor_pos - self._width
            case 'down':
                self.actor_pos = self.actor_pos + self._width
            case 'left':
                self.actor_pos = self.actor_pos - 1
            case 'right':
                self.actor_pos = self.actor_pos + 1
        
        self.state[self.actor_pos] = 1

        value = self.rewards[self.actor_pos] if self.actor_pos in self.rewards else np.array([-1, 0])
        if self.actor_pos == self.goal:
            return value, True
        else:
            return value, False

class PickUpAndDeliver(GridWorld):
    def __init__(self, initial_state, reward_function, state_function, actor_pos, height, width):
        self.state = initial_state
        self.reward = reward_function
        self.update_state = state_function
        self.actor_pos = actor_pos
        self._height = height
        self._width = width
        self._steps = 0

    def act(self, action):
        self._steps += 1
        if self._steps == 50:
            return np.array([-25,-50]), True
        if self.out_of_bounds(action):
            return np.array([-10, 0]), False
        match action:
            case 'up':
                self.actor_pos = self.actor_pos - self._width
            case 'down':
                self.actor_pos = self.actor_pos + self._width
            case 'left':
                self.actor_pos = self.actor_pos - 1
            case 'right':
                self.actor_pos = self.actor_pos + 1

        state, is_terminal = self.update_state(self.state, self.actor_pos) 
        previous_state = self.current_state()
        self.state = state
        value = self.reward(state, previous_state)

        if is_terminal:
            return value, True
        else:
            return value, False 

    def current_state(self):
        return self.state