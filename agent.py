import random
import gc

import numpy
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Input, LeakyReLU, Activation, BatchNormalization, Reshape, Conv2D, MaxPooling2D, Conv2DTranspose
# from keras.optimizers import Adam
from keras.optimizers.legacy import Adam, SGD
from collections import deque
from cube import Cube

tf.config.set_visible_devices([], 'GPU')


class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.memory = deque(maxlen=20000)
        self.tr_data = list()
        self.gamma = 0.97  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.state_shape = state_shape
        self.action_size = action_size
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_shape[1:])))
        # model.add(Reshape((9, 6, 6)))
        # model.add(Conv2D(filters=32, kernel_size=(2, 3)))
        # model.add(Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same'))
        # model.add(Conv2D(filters=32, kernel_size=(3, 3)))
        # model.add(Conv3D(filters=32, kernel_size=(3, 1)))
        # model.add(Conv3D(filters=32, kernel_size=(1, 3)))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))

        # model.add(MaxPooling2D(pool_size=(2, 2)))

        # model.add(Conv2D(filters=64, kernel_size=(3, 3)))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state.copy(), action, reward, next_state.copy(), done))

    def act(self, state)->(int, bool):
        if np.random.rand() <= self.epsilon:
            random_action = random.randrange(self.action_size)
            return random_action, True
        act_values = self.model.predict(Cube.reshape(state), verbose=0)
        return np.argmax(act_values[0]), False  # returns best action

    def replay(self, batch_size):
        minibatch1 = random.sample(self.memory, batch_size)
        minibatch2 = []
        if len(self.tr_data) > 0:
            minibatch2 = random.sample(self.tr_data, batch_size//2)
        minibatch = minibatch1 + minibatch2
        next_state_batch = np.array([next_state for state, action, reward, next_state, done in minibatch])
        state_batch = np.array([state for state, action, reward, next_state, done in minibatch])
        next_state_predict = self.model.predict(next_state_batch, verbose=0)
        target_f = self.model.predict(state_batch, verbose=0)
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                # print(f'Q: {target} + {self.gamma * np.amax(next_state_predict[i][0])}')
                target += (self.gamma * np.amax(next_state_predict[i][0]))
            # else:
            #     print(f'D: {target}')
            target_f[i][action] = target
        history = self.model.fit(state_batch, target_f, batch_size=16, epochs=1, verbose=0)
        return np.mean(history.history['loss'])

    def end_of_episode(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def add_experience(self, experience: list):
        self.memory.extend(experience)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def try_to_solve(self, cube: Cube, timeout=50):
        move_count = 0
        move_chain = []
        print(f'    Solving a cube {cube.puzzle_type}')
        cube.reset()
        cube.shuffle()
        cube.render()
        while move_count < timeout:
            prediction = self.model.predict(Cube.reshape(cube.current_state), verbose=0)
            best_action = np.argmax(prediction[0])
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
        for layer in self.model.layers:
            title = f'####   {layer.name}   ####'
            print()
            print(len(title) * '#')
            print(title)
            print(len(title) * '#')
            print(f'\n{layer.get_weights()}')
            # all_weights[layer.name] = layer.get_weights()
        # print(all_weights)

    def cleanup_trick(self):
        self.save('./model/weights')
        del self.model
        gc.collect()
        self.model = self._build_model()
        self.load('./model/weights')


