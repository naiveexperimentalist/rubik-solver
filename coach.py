import numpy as np
from cube import Cube


class Coach:
    @staticmethod
    def gen_tr_data(cube: Cube, episodes=10, depth=3):
        training_memory = list()
        for e in range(episodes):
            print(f'Training data generator, episode #{e}')
            current_depth = 0
            next_state = cube.solution_state.copy()
            while current_depth < depth:
                is_solved = cube.is_state_solved(next_state)
                move, previous_state = cube.simulate(next_state, -1)  # random move
                reversed_move = cube.get_move_back_index(move)
                reward = cube.reward_between_states(previous_state, next_state)
                # print(f'    zapisany ruch: {reversed_move} ({cube.possible_moves[reversed_move]["name"]}) REWARD: {reward}, IS_SOLVED: {is_solved}')
                training_memory.append((previous_state, reversed_move, reward, next_state, is_solved))
                next_state = previous_state.copy()
                current_depth += 1
        return training_memory

    @staticmethod
    def test(cube: Cube):
        cube.set_done_state()
        cube.render()
        actions = ["-f2", "-d1", "-d0"]
        for a in actions:
            for i, pm in enumerate(cube.possible_moves):
                if pm['name'] == a:
                    cube.move(i)
                    cube.render()
                    break
        # cube.move(cube.possible_moves[])
