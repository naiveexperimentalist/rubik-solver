import random
import numpy as np
from collections import deque
from cube import Cube
from model import RubikPolicy, RubikDataset

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
        self.memory = deque(maxlen=10000)
        self.tr_data = list()
        self.gamma = 0.6  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.04
        self.epsilon_decay = 0.9997
        self.learning_rate = 0.0075
        self.state_shape = state_shape
        self.action_size = action_size
        self.models = []
        self.model_cnt = 3
        for i in range(self.model_cnt):
            self.models.append(self._build_model())

    def _build_model(self):
        return RubikPolicy(self.state_shape, self.action_size)

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
            minibatch2 = random.sample(self.tr_data, batch_size)
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

    def replay2(self, batch_size):
        minibatch1 = random.sample(self.memory, batch_size)
        minibatch2 = []
        if len(self.tr_data) > 0:
            minibatch2 = random.sample(self.tr_data, batch_size//10)
        minibatch = minibatch1 + minibatch2
        next_state_batch = np.array([next_state for state, action, reward, next_state, done in minibatch])
        state_batch = np.array([state for state, action, reward, next_state, done in minibatch])
        next_state_predicts = []
        targets_f = []
        for m in self.models:
            next_state_predicts.append(m.mass_predict(next_state_batch))
            targets_f.append(m.mass_predict(state_batch))
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if self.model_cnt > 1:
                rnd1 = np.random.randint(self.model_cnt)
                rnd2 = np.random.randint(self.model_cnt)
            else:
                rnd1 = rnd2 = 0
            if not done:
                action = np.argmax(targets_f[rnd1][i])
                target += self.gamma * next_state_predicts[rnd2][i][action]
            targets_f[rnd1][i][action] = target
        history = []
        rnd = np.random.randint(self.model_cnt)
        h = np.mean(m.fit(state_batch, targets_f[rnd], batch_size=16, epochs=20, learning_rate=self.learning_rate))
        history.append(h)
        return np.mean(history)

    def replay3(self, batch_size):
        minibatch1 = random.sample(self.memory, batch_size)
        minibatch2 = []
        if len(self.tr_data) > 0:
            minibatch2 = random.sample(self.tr_data, batch_size//2)
        minibatch = minibatch1 + minibatch2
        next_state_batch = np.array([next_state for state, action, reward, next_state, done in minibatch])
        state_batch = np.array([state for state, action, reward, next_state, done in minibatch])
        next_state_predict = self.models[0].mass_predict(next_state_batch)
        target_f = self.models[0].mass_predict(state_batch)
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                action = np.argmax(target_f[i])
                target += self.gamma * next_state_predict[i][action]
            target_f[i][action] = target
        h = np.mean(self.models[0].fit(state_batch, target_f, batch_size=16, epochs=1, learning_rate=self.learning_rate))
        return h

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
