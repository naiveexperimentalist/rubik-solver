import random

import numpy as np
import hashlib


class Cube:
    def __init__(self, definition: dict, generate_pre_rewards=False):
        self.id = definition['id']
        self.puzzle_type = definition['puzzle_type']
        self.colors = sorted(list(set(definition['solution_state'])))
        self.color_to_index = {color: index for index, color in enumerate(self.colors)}
        self.solution_state = self.hot_encode(definition['solution_state'])
        self.initial_state = self.hot_encode(definition['initial_state'])
        self.current_state = self.hot_encode(definition['initial_state'])
        self.last_state = self.current_state
        self.last_move_index = -1  # <0 denotes the initial state (no last move)
        self.state_size = len(definition['solution_state'])
        self.wildcards = definition['wildcards']
        self.possible_moves = definition['moves']
        self.possible_moves_cnt = len(self.possible_moves)
        self.max_reward = 1000
        self.cc = ['\033[107m', '\033[42m', '\033[41m', '\033[44m', '\033[105m', '\033[103m']
        self.bc = ['\033[37m', '\033[92m', '\033[91m', '\033[94m', '\033[35m', '\033[33m']
        self.ee = '\033[0m'
        self.ccc = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8,
                    'j': 9, 'k': 10, 'l': 11, 'n': 12, 'o': 13, 'p': 14, 'q': 15, 'r': 16, 's': 17,
                    't': 36, 'u': 37, 'v': 38, 'x': 39, 'y': 40, 'z': 41, '*': 42, '^': 43, '>': 44}

        self.cube_dr = "      --------------------\n" + \
                       "     |\\aaaaaa\\bbbbbb\\cccccc\\ \n" + \
                       "     |t\\aaaaaa\\bbbbbb\\cccccc\\ \n" + \
                       "     |tt\\------\\------\\------\\ \n" + \
                       "     |tt|\\dddddd\\eeeeee\\ffffff\\ \n" + \
                       "     |\\t|u\\dddddd\\eeeeee\\ffffff\\ \n" + \
                       "     |x\\|uu\\------\\------\\------\\ \n" + \
                       "     |xx\\uu|\\gggggg\\hhhhhh\\iiiiii\\ \n" + \
                       "     |xx|\\u|v\\gggggg\\hhhhhh\\iiiiii\\ \n" + \
                       "     |\\x|y\\|vv -------------------- \n" + \
                       "     |*\\|yy\\vv|jjjjjj|kkkkkk|llllll|\n" + \
                       "     |**\\yy|\\v|jjjjjj|kkkkkk|llllll|\n" + \
                       "     |**|\\y|z\\|jjjjjj|kkkkkk|llllll|\n" + \
                       "      \\*|^\\|zz|--------------------|\n" + \
                       "       \\|^^\\zz|nnnnnn|oooooo|pppppp|\n" + \
                       "        \\^^|\\z|nnnnnn|oooooo|pppppp|\n" + \
                       "         \\^|>\\|nnnnnn|oooooo|pppppp|\n" + \
                       "          \\|>>|--------------------|\n" + \
                       "           \\>>|qqqqqq|rrrrrr|ssssss|\n" + \
                       "            \\>|qqqqqq|rrrrrr|ssssss|\n" + \
                       "             \\|qqqqqq|rrrrrr|ssssss|\n" + \
                       "               -------------------- \n"

        self.cube_flat = "\n" + \
                         "             +------------+\n" + \
                         "             |aaaabbbbcccc|\n" + \
                         "             |aaaabbbbcccc|\n" + \
                         "             |ddddeeeeffff|\n" + \
                         "             |ddddeeeeffff|\n" + \
                         "             |gggghhhhiiii|\n" + \
                         "             |gggghhhhiiii|\n" + \
                         "+------------+------------+------------+------------+\n" + \
                         "|GGGGHHHHIIII|jjjjkkkkllll|ttttuuuuvvvv|!!!!????####|\n" + \
                         "|GGGGHHHHIIII|jjjjkkkkllll|ttttuuuuvvvv|!!!!????####|\n" + \
                         "|TTTTUUUUVVVV|nnnnoooopppp|xxxxyyyyzzzz|$$$$%%%%&&&&|\n" + \
                         "|TTTTUUUUVVVV|nnnnoooopppp|xxxxyyyyzzzz|$$$$%%%%&&&&|\n" + \
                         "|XXXXYYYYZZZZ|qqqqrrrrssss|>>>><<<<^^^^|****(((())))|\n" + \
                         "|XXXXYYYYZZZZ|qqqqrrrrssss|>>>><<<<^^^^|****(((())))|\n" + \
                         "+------------+------------+------------+------------+\n" + \
                         "             |JJJJKKKKLLLL|\n" + \
                         "             |JJJJKKKKLLLL|\n" + \
                         "             |NNNNOOOOPPPP|\n" + \
                         "             |NNNNOOOOPPPP|\n" + \
                         "             |QQQQRRRRSSSS|\n" + \
                         "             |QQQQRRRRSSSS|\n" + \
                         "             +------------+"

        ff = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
              'j', 'k', 'l', 'n', 'o', 'p', 'q', 'r', 's',
              't', 'u', 'v', 'x', 'y', 'z', '>', '<', '^',
              '!', '?', '#', '$', '%', '&', '*', '(', ')',
              'G', 'H', 'I', 'T', 'U', 'V', 'X', 'Y', 'Z',
              'J', 'K', 'L', 'N', 'O', 'P', 'Q', 'R', 'S']
        self.fff = {ch: i for i, ch in enumerate(ff)}
        self.known_states_rewards = {}
        if generate_pre_rewards:
            states_rewards = list()
            self.generate_high_quality_actions(self.solution_state, max_depth=2, expandable=states_rewards)
            self.known_states_rewards = {sr['hash']: {'reward': sr['reward'], 'state': sr['state']} for sr in states_rewards}

    def set_state(self, s):
        self.current_state = s.copy()
        self.last_move_index = -1

    def set_done_state(self):
        self.set_state(self.solution_state)

    def reset(self):
        self.set_state(self.initial_state)

    def shuffle(self):
        probability = np.random.randint(5)
        # from time to time shuffle to the well-known state
        if probability == 3:
            random_key, random_val = random.sample(self.known_states_rewards.items(), 1)[0]
            self.set_state(random_val['state'])
            return

        self.set_done_state()
        for i in range(20):
            self.move(-1)
        self.last_move_index = -1

    def is_solved(self):
        return self.is_state_solved(self.current_state)

    def is_state_solved(self, state):
        return self.distance_to_solution(state) <= self.wildcards + 0.5

    def simulate(self, cur_state, move_index):
        # move_index >= 0 ===> wskazany konkretny ruch
        # move_index < 0  ===> prośba o wykonanie kolejnego randomowego ruchu
        if move_index < 0:  # make it random
            move_index = np.random.randint(self.possible_moves_cnt)
        move = self.possible_moves[move_index]
        return move_index, cur_state[move['map']].copy()

    def move(self, move_index: int):
        move_done, new_state = self.simulate(self.current_state, move_index)
        self.last_state = self.current_state
        self.current_state = new_state
        self.last_move_index = move_done

    # def move_back(self):
    #     if self.last_move_index < 0:
    #         print("Can't make a move back, since no move has been made yet")
    #         pass  # no move so far, so move-back not possible
    #     reverse_last_move_name = self.get_move_back_index(self.last_move_index)
    #     self.move(reverse_last_move_name)

    def current_distance_to_solution(self) -> int:
        return self.distance_to_solution(self.current_state)

    def distance_to_solution(self, state) -> int:
        return np.sum(np.abs(self.solution_state-state))//2

    def reward_between_states(self, prev_s, next_s):
        dist1 = self.distance_to_solution(prev_s)
        dist2 = self.distance_to_solution(next_s)
        if dist2 == 0:
            reward = self.max_reward
        else:
            next_state_hash = self.hash(next_s)
            if dist1 - dist2 > 0 and self.hash(next_s) in self.known_states_rewards:
                print(f'HIGH QUALITY MOVE DETECTED, REWARD: {self.known_states_rewards[next_state_hash]["reward"]}')
                reward = self.known_states_rewards[next_state_hash]['reward']
            else:
                reward = dist1 - dist2
                # reward = 0
        # print(reward)
        return reward

    def reward(self):
        return self.reward_between_states(self.last_state, self.current_state)

    def get_move_back_index(self, move_forward_index):
        if move_forward_index < 0:
            return -1
        if self.possible_moves[move_forward_index]['reversed']:  # last move was backward
            return move_forward_index - 1
        return move_forward_index + 1

    def get_last_move_name(self):
        if self.last_move_index < 0:
            return '-'
        return self.possible_moves[self.last_move_index]['name']

    def hot_encode1(self, state):
        return np.array([self.color_to_index[c] for c in state])

    def hot_decode1(self, state):
        return [self.colors[i] for i in state]

    def hot_encode(self, cube_state):
        one_hot_encoded_state = []
        for color in cube_state:
            one_hot_vector = np.zeros(len(self.colors))
            one_hot_vector[self.color_to_index[color]] = 1
            one_hot_encoded_state.append(one_hot_vector)
        return np.array(one_hot_encoded_state)

    def hot_decode(self, state):
        m = np.where(state == 1)
        decoded_state = [self.colors[i] for i in m[1]]
        return decoded_state

    def render(self):
        s = self.hot_decode(self.current_state)
        draw_string = self.cube_dr
        for key, item in self.ccc.items():
            clr = s[item]
            clr_i = self.color_to_index[clr]
            draw_string = draw_string.replace(key, f'{self.cc[clr_i]}{self.bc[clr_i]}{clr}{self.ee}')
        draw_string2 = self.cube_flat
        for key, item in self.fff.items():
            clr = s[item]
            clr_i = self.color_to_index[clr]
            draw_string2 = draw_string2.replace(key, f'{self.cc[clr_i]}{self.bc[clr_i]} {self.ee}')
        print()
        # print(draw_string)
        print(draw_string2)
        print('\n\n\n')
        # print(s)

    def hash(self, state):
        return hashlib.md5(f'{self.hot_decode(state)}'.encode("utf-8")).hexdigest()

    def agent_key(self):
        return hashlib.md5(f'{self.hot_decode(self.solution_state)}__{self.wildcards}'.encode("utf-8")).hexdigest()

    @staticmethod
    def reshape(s):
        return np.reshape(s, s.shape)

    def to_2d(self, s):
       return np.reshape(self.wall_cnt, self.wall_size, -1, 'C')

    def to_1d(self, s):
        return np.reshape(self.wall_cnt * self.wall_size, -1, 'C')

    def generate_high_quality_actions(self, cur_state, max_depth=3, cur_depth=1, expandable=[], extra_info=0):
        for i in range(self.possible_moves_cnt):
            move_done, new_state = self.simulate(cur_state, i)
            action = {
                'hash': self.hash(cur_state),
                'reward': self.max_reward//(2*cur_depth),
                'state': cur_state
            }
            expandable.append(action)
            if cur_depth < max_depth:
                self.generate_high_quality_actions(new_state, max_depth, cur_depth+1, expandable, extra_info=i)




