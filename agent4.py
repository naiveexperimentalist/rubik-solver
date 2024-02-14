import random
import numpy as np
from collections import deque
from cube import Cube
from model4 import RubikPolicy, RubikDataset

import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.memory_size = 10000
        self.memory = deque(maxlen=self.memory_size)
        self.tr_data = list()
        self.gamma = 0.66  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.04
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0001
        self.state_shape = state_shape
        self.action_size = action_size
        self.models = []
        self.model_cnt = 3
        for i in range(self.model_cnt):
            self.models.append(self._build_model())

    def _build_model(self):
        m = RubikPolicy(self.state_shape, self.action_size)
        m.initialize_weights()
        return m

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state.copy(), action, reward, next_state.copy(), done))

    def act(self, state) -> (int, bool):
        if np.random.rand() <= self.epsilon:
            random_action = np.random.randint(self.action_size)
            return random_action, True
        act_values = self.predict(Cube.reshape(state))
        return np.argmax(act_values), False  # returns best action

    def replay(self, batch_size):
        minibatch1 = random.sample(self.memory, batch_size)
        minibatch2 = []
        if len(self.tr_data) > 0:
            minibatch2 = random.sample(self.tr_data, batch_size//2)
        minibatch = minibatch1 + minibatch2
        history = []
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if self.model_cnt > 1:
                model1 = self.models[np.random.randint(self.model_cnt)]
                model2 = self.models[np.random.randint(self.model_cnt)]
            else:
                model1 = model2 = self.models[0]
            if not done:
                action_from_m1 = np.argmax(model1.predict(next_state))
                target += self.gamma * model2.predict(next_state)[action_from_m1]
            target_f = model1.predict(state)
            target_f[action] = target
            h = model1.single_fit(state, target_f, epochs=1, learning_rate=self.learning_rate, verbose=0)
            history.append(h)
        return np.mean(history)

    def end_of_episode(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def add_experience(self, experience: list):
        self.memory.extend(experience)

    # def load(self, name):
    #     self.model.load_weights(name)
    #
    # def save(self, name):
    #     self.model.save_weights(name)

    def try_to_solve(self, cube: Cube, timeout=50):
        move_count = 0
        move_chain = []
        print(f'    Solving a cube {cube.puzzle_type}')
        cube.reset()
        cube.shuffle()
        cube.render()
        while move_count < timeout:
            prediction = self.predict(Cube.reshape(cube.current_state))
            print('!!!!!!!!!   NEXT MOVE    !!!!!!!!!!')
            print(cube.hot_decode(cube.current_state))
            print(np.round(prediction, 2))
            best_action = np.argmax(prediction)
            cube.move(best_action)
            move_chain.append(cube.get_last_move_name())
            if cube.is_solved():
                print(move_chain)
                print('DONE!!!')
                return
            move_count += 1
        cube.render()
        print(move_chain)
        print(f'Exiting after trying {timeout} moves.')

    def show_network(self):
        # all_weights = {}
        # print(self.model.layers)
        for layer in self.models[-1].layers:
            title = f'####   {layer.name}   ####'
            print()
            print(len(title) * '#')
            print(title)
            print(len(title) * '#')
            print(f'\n{layer.get_weights()}')
            # all_weights[layer.name] = layer.get_weights()
        # print(all_weights)

    def predict(self, s):
        predictions = np.array([m.predict(s) for m in self.models])
        return np.max(predictions, axis=0)

    def model_summary(self):
        self.models[0].summary()

    def model_weights(self):
        for m in self.models:
            m.print_weights()
