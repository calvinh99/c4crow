from .base_player import Player
import numpy as np
import random
import math
import time
import sys
import c4crow.c4_engine as c4

class MCTSPlayer(Player):
    def __init__(self, n_iterations=10000, xray=False):
        self.n_iterations = n_iterations
        self.xray = xray
        self.C = np.sqrt(2)  # exploration coefficient

    def make_move(self, board: np.ndarray, piece: int) -> int:
        root = board
        root_key = c4.make_board_hashable(root)
        root_piece = piece
        state_dict = {}

        root_children = {a: c4.make_board_hashable(c4.drop_piece(root, root_piece, a)) 
                         for a in c4.get_available_cols(root)}

        start_time = time.time()
        last_updated_time = start_time - 10
        update_interval_seconds = 0.5
        last_lines_printed = 0

        for i in range(self.n_iterations):
            traversal = self.select(root, root_piece, state_dict)
            expand_state = c4.string_to_board(traversal[-1][0])
            current_piece = traversal[-1][1]

            if expand_state is None:
                continue

            reward = self.simulate(expand_state, current_piece, root_piece)
            self.backprop(traversal, state_dict, reward, root_piece)

            if self.xray and (time.time() - last_updated_time) >= update_interval_seconds:
                last_lines_printed = self.update_display(i + 1, start_time, root_children, root_key, state_dict, last_lines_printed)
                last_updated_time = time.time()

        if self.xray:
            self.update_display(self.n_iterations, start_time, root_children, root_key, state_dict, last_lines_printed)

        root_action_metrics = [[a, state_dict[child_key]['N']] for a, child_key in root_children.items() if child_key in state_dict]
        return max(root_action_metrics, key=lambda x: x[1])[0]

    def select(self, root, root_piece, state_dict):
        traversal = []
        parent = root
        current_piece = root_piece

        while True:
            parent_key = c4.make_board_hashable(parent)
            for a in c4.get_available_cols(parent):
                child = c4.drop_piece(parent, current_piece, a)
                child_key = c4.make_board_hashable(child)
                if child_key not in state_dict or state_dict[child_key]['N'] == 0:
                    traversal.append([parent_key, c4.get_opponent_piece(current_piece)])
                    traversal.append([child_key, current_piece])
                    return traversal

            max_a, max_UCT = None, float("-inf")
            for a in c4.get_available_cols(parent):
                child = c4.drop_piece(parent, current_piece, a)
                child_key = c4.make_board_hashable(child)
                UCT = self.calc_UCT(parent_key, child_key, state_dict)
                if UCT > max_UCT:
                    max_a = a
                    max_UCT = UCT

            traversal.append([parent_key, c4.get_opponent_piece(current_piece)])
            parent = c4.drop_piece(parent, current_piece, max_a)
            current_piece = c4.get_opponent_piece(current_piece)

    def simulate(self, state, current_piece, root_piece):
        while True:
            game_status = c4.check_win(state, current_piece)
            if game_status == "win":
                return 1 if current_piece == root_piece else -1
            elif game_status == "draw":
                return 0
            a = random.choice(c4.get_available_cols(state))
            state = c4.drop_piece(state, current_piece, a)
            current_piece = c4.get_opponent_piece(current_piece)

    def backprop(self, traversal, state_dict, reward, root_piece):
        for state_key, piece in traversal:
            if state_key not in state_dict:
                state_dict[state_key] = {'N': 1, 'W': 0, 'WR': 0}
            else:
                state_dict[state_key]['N'] += 1
            actual_reward = reward if piece == root_piece else -reward
            state_dict[state_key]['W'] += actual_reward
            state_dict[state_key]['WR'] = state_dict[state_key]['W'] / state_dict[state_key]['N']

    def calc_UCT(self, parent_key, child_key, state_dict):
        exploit = state_dict[child_key]['WR']
        explore = math.sqrt(math.log(state_dict[parent_key]['N']) / state_dict[child_key]['N'])
        return exploit + self.C * explore

    def update_display(self, i, start_time, root_children, root_key, state_dict, last_lines_printed):
        elapsed_time = time.time() - start_time
        iterations_per_second = i / elapsed_time if elapsed_time > 0 else 0

        if last_lines_printed > 0:
            sys.stdout.write(f"\033[{last_lines_printed}A")
            sys.stdout.write("\033[J")

        print(f"MCTS Progress: {i}/{self.n_iterations} | {iterations_per_second:.2f} it/s")
        print("Action metrics:")
        lines_printed = 2

        for a, child_key in root_children.items():
            if child_key in state_dict:
                child_stats = state_dict[child_key]
                UCT = self.calc_UCT(root_key, child_key, state_dict)
                print(f"Action {a}: N={child_stats['N']}, WR={child_stats['WR']:.2f}, UCT={UCT:.2f}")
                lines_printed += 1

        print()
        lines_printed += 1
        return lines_printed
