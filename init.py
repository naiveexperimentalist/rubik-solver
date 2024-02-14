import pandas as pd
import numpy as np
import json
from cube import Cube
from agent2 import DQNAgent
from coach import Coach
import time

# load puzzle types
df_puzzle_info = pd.read_csv('./data/puzzle_info.csv')
puzzle_types = dict()
for i, row in df_puzzle_info.iterrows():
    print(f'#{i} --> {row["puzzle_type"]}')
    moves_raw = json.loads(row['allowed_moves'].replace("'", '"'))
    # defined moves

    moves = []
    for item in moves_raw.items():
        indices = np.array(item[1])
        # add forward move
        moves.append({'name': item[0], 'map': np.array(indices), 'reversed': False})
        indices_of_indices = range(len(indices))
        reversed_map = np.array([np.argwhere(indices==i)[0][0] for i in indices_of_indices])
        # add backward move
        moves.append({'name': f'-{item[0]}', 'map': reversed_map, 'reversed': True})
    puzzle_types[row['puzzle_type']] = moves

puzzle_agents = dict()
# load puzzles
df_puzzles = pd.read_csv('./data/puzzles.csv')
# print(df_puzzles)
puzzles = []
for i, row in df_puzzles.iterrows():
    print(f'initializing cube {row["puzzle_type"]}  {i+1}/{df_puzzles.shape[0]}  ...', end=' ')
    solution_state = row['solution_state'].split(';')
    initial_state = row['initial_state'].split(';')
    puzzle = [{item[0]: item[1]} for item in moves_raw.items()]
    puzzles.append(
        Cube(definition={'id': row['id'], 'puzzle_type': row['puzzle_type'], 'solution_state': np.array(solution_state),
                         'initial_state': np.array(initial_state), 'wildcards': int(row['num_wildcards']),
                         'moves': puzzle_types[row['puzzle_type']]}, generate_pre_rewards=(i==65)))
    print('    done')
    agent_key = puzzles[-1].agent_key()
    if i == 65 and agent_key not in puzzle_agents:
        state_shape = Cube.reshape(puzzles[-1].initial_state).shape
        action_size = len(puzzle_types[row["puzzle_type"]])
        print(f'Creating an agent for {row["puzzle_type"]}')
        puzzle_agents[agent_key] = DQNAgent(state_shape, action_size)

puzzle = puzzles[65]
# Coach.test(puzzle)

agent = puzzle_agents[puzzle.agent_key()]
training_data = Coach.gen_tr_data(puzzle, episodes=40, depth=10)
agent.tr_data = training_data
# agent.add_experience(training_data)

print(f'\n\n############ LEARNING HOW TO SOLVE {puzzle.puzzle_type} ############\n')
print(f'allowed wildcards: {puzzle.wildcards}')
print(f'possible moves count: {puzzle.possible_moves_cnt}')
print(f'agent key: {puzzle.agent_key}')
print(f'state size: {puzzle.state_size}')
agent.model_summary()
print('\n\n')
# Learning loop
EPISODES = 2500
EPISODE_SIZE = 150
batch_size = 128

for e in range(EPISODES):
    t1 = time.perf_counter()
    puzzle.reset()
    puzzle.shuffle()
    state = puzzle.current_state
    episode_loss = 0
    random_moves = 0
    thought_moves = 0
    for t in range(EPISODE_SIZE):  # max step count within a single episode
        action, was_random = agent.act(state)
        if was_random:
            random_moves += 1
        else:
            thought_moves += 1
        puzzle.move(action)
        next_state = puzzle.current_state.copy()
        reward = puzzle.reward()
        done = puzzle.is_solved()
        agent.remember(state, action, reward, next_state, done)
        state = next_state.copy()
        if done:
            print(f"SOLVED. Episode: {e}/{EPISODES}, Step: {t}")
            break
        if batch_size < len(agent.memory):
            loss = agent.replay2(batch_size)
            episode_loss += loss
    t2 = time.perf_counter()
    print(f'END OF EPISODE #{e}/{EPISODES} (took: {t2-t1}s). Loss = {episode_loss/EPISODE_SIZE}, Random_moves: {random_moves}, Thought moves: {thought_moves}, Epsilon: {agent.epsilon}')
    agent.end_of_episode()
    if (e+1) % 10 == 0:
        agent.try_to_solve(puzzle, timeout=30)
