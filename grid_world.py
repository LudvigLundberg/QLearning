import numpy as np

class GridWorld:
    def __init__(self, board, rewards, start, goal, height, width):
        self.state = board
        self.rewards = rewards
        self.actions = ['up','down','right','left']
        self.actor_pos = start
        self.goal = goal
        self._height = height
        self._width = width

    def current_state(self):
        return tuple(self.state)

    def act(self, action):
        self.state[self.actor_pos] = 0
        if self.out_of_bounds(action):
            return np.array([-10]), True
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

        value = self.rewards[self.actor_pos] if self.actor_pos in self.rewards else np.array([-1])
        if self.actor_pos == self.goal:
            return value, True
        else:
            return value, False
    
    def out_of_bounds(self, action):
        match action:
            case 'up':
                return self.actor_pos < self._width
            case 'down':
                return self.actor_pos >= len(self.state) - self._width
            case 'left':
                return self.actor_pos % self._width == 0
            case 'right':
                return self.actor_pos % self._width == self._width - 1