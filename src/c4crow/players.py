import os
import random
import math
import sys
import time
from pprint import pprint
from typing import Tuple
from collections import defaultdict

import numpy as np
import torch

import c4crow.c4_engine as c4
from c4crow.models.DQN import DQN, DQN2
from c4crow.models.PolicyNet import PolicyCNN

"""
All players take in some number of inputs and output the column index to drop the piece in.
"""

# ---------------------------------------------------
# Real Player
# ---------------------------------------------------

def real_player(board: np.ndarray, piece: int, **kwargs) -> int:
    while True:
        try:
            col = int(input(f"Player {piece}, enter your move (0-6): "))
            if col in c4.get_available_cols(board):
                return col
            else:
                print("Invalid move. Please choose an available column.")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 6.")

# ---------------------------------------------------
# Random Player
# ---------------------------------------------------

def random_player(board: np.ndarray, piece: int, **kwargs) -> int:
    available_cols = c4.get_available_cols(board)
    return random.choice(available_cols)

# ---------------------------------------------------
# Policy Net Player
# ---------------------------------------------------

def get_policy_net_player(path_to_weights: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PolicyCNN(c4.N_COLS)
    model.to(device)
    if path_to_weights is not None and os.path.exists(path_to_weights):
        state_dict = torch.load(path_to_weights, map_location=device)
        model.load_state_dict(state_dict)
    
    def policy_net_player(board: np.ndarray, training=False):
        available_cols = c4.get_available_cols(board)
        state = torch.tensor(board, dtype=torch.float32).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
        probs = model(state)
        mask = torch.zeros_like(probs)
        mask[0, available_cols] = 1
        masked_probs = probs * mask
        masked_probs = masked_probs / masked_probs.sum() # normalize again
        if training:
            action = torch.multinomial(masked_probs, 1).item()
            log_prob = torch.log(masked_probs[:, action])
            return action, log_prob
        else:
            action = torch.argmax(masked_probs).cpu().item()
            return action
    
    return policy_net_player, model

# ---------------------------------------------------
# DQN Player
# ---------------------------------------------------

def get_dqn_player(
    path_to_weights: str,
    architecture: str,
    eps_start=0.9,
    eps_end=0.05,
    eps_steps=2000
):
    # initialize the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if architecture == "DQN":
        model = DQN(c4.N_COLS).to(device)
    elif architecture == "DQN2":
        model = DQN2(c4.N_COLS).to(device)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    if path_to_weights is not None and os.path.exists(path_to_weights):
        state_dict = torch.load(path_to_weights, map_location=device)
        model.load_state_dict(state_dict)

    def dqn_player(
        board: np.ndarray, piece: int,
        training=False,
        steps_done=0
    ):
        available_cols = c4.get_available_cols(board)
        input_state = torch.tensor(board, dtype=torch.float, device=device).unsqueeze(dim=0).unsqueeze(dim=0)
        eps = random.random() # btwn 0 and 1

        if training:
            eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1 * steps_done / eps_steps)
        else:
            eps_threshold = 0
        
        # epsilon-greedy policy
        if eps > eps_threshold: # if training=False, always exploit
            with torch.no_grad():
                state_action_values = model(input_state)[0].cpu().numpy() # why is output shape (1, N_COLS)? why not just (N_COLS,)?
                state_action_values = [state_action_values[col] for col in available_cols] # get only available probs
                argmax_action = np.argmax(state_action_values) # get index of max val
                col_idx = available_cols[argmax_action] # get the col_idx corresponding to idx
                return col_idx
        else:
            return random.choice(available_cols) # exploration
        
    return dqn_player, model


# ---------------------------------------------------
# Minimax Player
# ---------------------------------------------------

MINIMAX_WIN_SCORE = 1000000 # don't set to inf, otherwise can't distinguish between earlier and later wins

def get_board_score(board: np.ndarray, piece: int):
    # helper function for minimax player, but possibly could be used in other players
    if c4.check_win(board, piece) == "win":
        return MINIMAX_WIN_SCORE
    elif c4.check_win(board, piece.get_opponent_piece()) == "win":
        return -MINIMAX_WIN_SCORE

    score = 0

    # add number of cetner cells belonging to piece
    center_col_idx = c4.N_COLS // 2
    score += 3 * sum(board[:, center_col_idx] == piece._value_) # reward as much as a 3 in a row

    # piece connections of our piece and opponent
    n_connections = c4.count_piece_connections(board, piece)
    n_opponent_connections = c4.count_piece_connections(board, piece.get_opponent_piece())

    for n, count in n_connections.items():
        score += n * count
    
    for n, count in n_opponent_connections.items():
        score -= n * count # punish for piece connections of opponent
    
    return score
    
def get_minimax_player(max_depth: int = 4, xray: bool = False) -> int:
    def minimax_player(board: np.ndarray, piece: int) -> int:
        # so for minimax
        # we have maxxing and minning
        # for each state we serarch through all possible moves
        # We get a score back
        # we choose max score if maxxing, min score if not
        # this means for us and our opponent we are making the "best possible" moves.
        # so our move is the best possibe response to the opponent's best possible move
        # then opponent chooses their best possible response, and on and on
        # until finally my real move is best possible move accounting for all of these future moves.
        # Then we choose the move that has max score

        def minimax_tree_search(
            board: np.ndarray, piece: int, 
            depth: int, maximizing: bool,
            # alpha: float, beta: float
        ) -> Tuple[int, float]:
            available_cols = c4.get_available_cols(board)

            if depth == 0 or c4.check_win(board, piece.get_opponent_piece()) or len(available_cols) == 0: # need to get opponent piece, because opponent piece was the one that just made a move, not current piece, current piece has yet to make a move.
                return None, get_board_score(board, piece if maximizing else piece.get_opponent_piece())

            best_col = random.choice(available_cols)
            if maximizing:
                best_score = float("-inf")
                for col in available_cols:
                    new_board = c4.drop_piece(board, piece, col)
                    _, score = minimax_tree_search(
                        new_board, piece.get_opponent_piece(), depth - 1, False,
                        # alpha, beta
                    )
                    if score > best_score:
                        best_score, best_col = score, col
                    # alpha = max(alpha, best_score) # if this move results in good score (high), update alpha (our assured minimum score)
                    # if alpha >= beta:
                    #     break
            else:
                best_score = float("inf")
                for col in available_cols:
                    new_board = c4.drop_piece(board, piece, col)
                    _, score = minimax_tree_search(
                        new_board, piece.get_opponent_piece(), depth - 1, True,
                        # alpha, beta
                    )
                    if score < best_score:
                        best_score, best_col = score, col
                    # beta = min(beta, best_score) # if this move results in good score (low), update beta (our assured maximum score)
                    # if alpha >= beta:
                    #     break

            return best_col, best_score

        best_col, score = minimax_tree_search(
            board, piece, max_depth, True, 
            # float("-inf"), float("inf")
        )
        if xray:
            print(f"[X-Ray] For Player {piece}, the estimated future score of dropping piece in {best_col} is {score}.")
        return best_col
    
    return minimax_player

# ---------------------------------------------------
# Monte Carlo Tree Search Player
# ---------------------------------------------------

def get_mcts_player(n_iterations=10000, xray=False):

    C = np.sqrt(2) # exploration coefficient

    def calc_UCT(parent_key, child_key, state_dict):
        exploit = state_dict[child_key]['WR']
        explore = math.sqrt(math.log(state_dict[parent_key]['N']) / state_dict[child_key]['N'])
        UCT = exploit + C * explore
        return UCT
    
    def backprop(traversal, state_dict, reward, root_piece):
        for state_key, piece in traversal:
            if state_key not in state_dict:
                state_dict[state_key] = {'N': 1, 'W': 0, 'WR': 0}
            else:
                state_dict[state_key]['N'] += 1
            actual_reward = reward if piece == root_piece else -reward
            state_dict[state_key]['W'] += actual_reward
            state_dict[state_key]['WR'] = state_dict[state_key]['W'] / state_dict[state_key]['N']

    def mcts_player(board: np.ndarray, piece: int) -> int:
        root = board
        root_key = c4.make_board_hashable(root)
        root_piece = piece
        state_dict = {} # state -> visit count, win count, win rate

        root_children = {}
        for a in c4.get_available_cols(root):
            child = c4.drop_piece(root, root_piece, a)
            child_key = c4.make_board_hashable(child)
            root_children[a] = child_key

        start_time = time.time()
        last_updated_time = start_time - 10
        update_interval_seconds = 0.5
        min_wait_time_per_iter = 0 # for debugging
        last_lines_printed = 0

        def update_display(i):
            nonlocal last_lines_printed, last_updated_time
            current_time = time.time()
            elapsed_time = current_time - start_time
            iterations_per_second = i / elapsed_time if elapsed_time > 0 else 0

            # Clear previous lines
            if last_lines_printed > 0:
                sys.stdout.write(f"\033[{last_lines_printed}A")  # Move cursor up
                sys.stdout.write("\033[J")  # Clear from cursor to end of screen

            progress = f"MCTS Progress: {i}/{n_iterations} | {iterations_per_second:.2f} it/s"
            print(progress)
            
            print("Action metrics:")
            lines_printed = 2  # Count for progress and "Action metrics:" lines
            for a, child_key in root_children.items():
                if child_key in state_dict:
                    child_stats = state_dict[child_key]
                    UCT = calc_UCT(root_key, child_key, state_dict)
                    print(f"Action {a}: N={child_stats['N']}, WR={child_stats['WR']:.2f}, UCT={UCT:.2f}")
                    lines_printed += 1
            print(); lines_printed += 1 # fixes terminal print freeze issue
            time.sleep(min_wait_time_per_iter)
            last_updated_time = current_time
            last_lines_printed = lines_printed

        for i in range(n_iterations):
            # select phase
            # choose the best child node based on Upper Confidence Bound formula
            traversal = [] # state, action, piece
            parent = root
            current_piece = root_piece
            expand_state = None
            while True:            
                # check if parent is leaf
                parent_key = c4.make_board_hashable(parent)
                for a in c4.get_available_cols(parent):
                    child = c4.drop_piece(parent, current_piece, a)
                    child_key = c4.make_board_hashable(child)
                    if child_key not in state_dict or state_dict[child_key]['N'] == 0: # parent is leaf state
                        traversal.append([parent_key, c4.get_opponent_piece(current_piece)]) # seriously need to add visualization (step by step) to validate this
                        traversal.append([child_key, current_piece])
                        expand_state = child
                        current_piece = c4.get_opponent_piece(current_piece)
                        break
                if expand_state is not None: break

                # parent is not leaf, select highest UCT
                max_a, max_UCT = None, float("-inf")
                for a in c4.get_available_cols(parent):
                    child = c4.drop_piece(parent, current_piece, a)
                    child_key = c4.make_board_hashable(child)
                    UCT = calc_UCT(parent_key, child_key, state_dict)
                    if UCT > max_UCT:
                        max_a = a
                        max_UCT = UCT
                traversal.append([parent_key, current_piece])
                parent = c4.drop_piece(parent, current_piece, max_a)
                current_piece = c4.get_opponent_piece(current_piece)

            if expand_state is None:
                continue # backpropped already, continue to next iteration. Is this necessary? Should we just exit from func?

            # simulate
            curr_state = expand_state
            while True:
                a = random.choice(c4.get_available_cols(curr_state))
                curr_state = c4.drop_piece(curr_state, current_piece, a)
                game_status = c4.check_win(curr_state, current_piece)
                if game_status == "win" and current_piece == root_piece:
                    backprop(traversal, state_dict, 1, root_piece); break
                elif game_status == "win" and current_piece != root_piece:
                    backprop(traversal, state_dict, -1, root_piece); break
                elif game_status == "draw":
                    backprop(traversal, state_dict, 0, root_piece); break
                current_piece = c4.get_opponent_piece(current_piece)

            if xray and (time.time() - last_updated_time) >= update_interval_seconds:
                update_display(i + 1)

        if xray:
            update_display(n_iterations)

        root_action_metrics = []  # action, visit count
        for a, hashable_child in root_children.items():
            if hashable_child in state_dict:
                root_action_metrics.append([a, state_dict[hashable_child]['N']])

        return max(root_action_metrics, key=lambda x: x[1])[0]
                    
    return mcts_player