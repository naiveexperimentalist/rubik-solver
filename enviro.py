from abc import ABC

from cube import Cube
from rl.core import Env


class Enviro(Cube, Env):
    def __init__(self, definition: dict):
        super().__init__(definition, agent_key='xyz')

    def step(self, action):
        self.move(action)
        return self.current_state, self.reward(), self.is_solved(), {}

    def reset(self):
        super().reset()
        return self.current_state  # return initial observation

    def render(self, mode='human', close=False):
        super().render()

    def close(self):
        pass

    def seed(self):
        pass

    def configure(self):
        pass
