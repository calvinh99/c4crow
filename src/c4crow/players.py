import os
import random
import math
from pprint import pprint
from typing import Tuple
from collections import defaultdict

import numpy as np
import torch

import c4crow.c4_engine as c4
from c4crow.DQN import DQN, DQN2
from c4crow.PolicyNet import PolicyCNN

"""
All players take in some number of inputs and output the column index to drop the piece in.
"""

# ---------------------------------------------------
# Real Player
# ---------------------------------------------------

def real_player(board: np.ndarray, piece: c4.Pieces, **kwargs) -> int:
    while True:
        try:
            col = int(input(f"Player {piece.display_str}, enter your move (0-6): "))
            if col in c4.get_available_cols(board):
                return col
            else:
                print("Invalid move. Please choose an available column.")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 6.")

# ---------------------------------------------------
# Random Player
# ---------------------------------------------------

def random_player(board: np.ndarray, piece: c4.Pieces, **kwargs) -> int:
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
        board: np.ndarray, piece: c4.Pieces,
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

def get_board_score(board: np.ndarray, piece: c4.Pieces):
    # helper function for minimax player, but possibly could be used in other players
    if c4.check_win(board, piece):
        return MINIMAX_WIN_SCORE
    elif c4.check_win(board, piece.get_opponent_piece()):
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
    def minimax_player(board: np.ndarray, piece: c4.Pieces) -> int:
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
            board: np.ndarray, piece: c4.Pieces, 
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
            print(f"[X-Ray] For Player {piece.display_str}, the estimated future score of dropping piece in {best_col} is {score}.")
        return best_col
    
    return minimax_player

# ---------------------------------------------------
# Monte Carlo Tree Search Player
# ---------------------------------------------------

def get_mcts_player(n_iterations=10, xray=False):

    c = np.sqrt(2) # exploration coefficient

    def get_unexplored_edges(node, state_action_tree): # it's a leaf node if there are any unexplored edges/child nodes or if it's a terminal node
        available_edges = c4.get_available_cols(node)
        state = c4.make_board_hashable(node)
        unexplored_edges = []
        for edge in available_edges:
            state_action = tuple(list(state) + [edge])
            if state_action not in state_action_tree:
                unexplored_edges.append(edge)
        return unexplored_edges
    
    def backprop(terminal_node, piece, traversal, state_action_tree):
        if len(c4.get_available_cols(terminal_node)) == 0:
            terminal_value = 0
        elif c4.check_win(terminal_node, piece):
            terminal_value = 1
        else:
            terminal_value = -1

        for state, action in traversal:
            if state not in state_action_tree:
                state_action_tree[state] = {'N': 1}
            else:
                state_action_tree[state]['N'] += 1
            if action not in state_action_tree[state]:
                state_action_tree[state][action] = {'N': 1, 'Q': terminal_value}
            else:
                state_action_tree[state][action]['N'] += 1
                state_action_tree[state][action]['Q'] += terminal_value

    def mcts_player(board: np.ndarray, piece: c4.Pieces) -> int:
        root_node = board
        state_action_tree = {} # 'state' -> dict of 'action's and 'N', 'action' -> dict of 'Q' and 'N'

        for i in range(n_iterations):
            # select phase
            # choose the best child node based on Upper Confidence Bound formula
            traversal = [] # state, expand_edge, piece
            selected_node = root_node
            temp_piece = piece
            expand_node = None
            while True:
                if c4.is_terminal(selected_node) and len(traversal) > 0:
                    backprop(selected_node, piece, traversal, state_action_tree)
                    break
                
                state = c4.make_board_hashable(selected_node)
                unexplored_edges = get_unexplored_edges(selected_node, state_action_tree)
                if len(unexplored_edges) > 0:
                    expand_edge = random.choice(unexplored_edges)
                    expand_node = c4.drop_piece(selected_node, temp_piece, expand_edge)
                    traversal.append((state, expand_edge))
                    break

                available_edges = c4.get_available_cols(selected_node)
                edge_values = []
                for edge in available_edges:
                    state_action = tuple(list(state) + [edge])
                    UCB = (
                        state_action_tree[state][edge]['Q'] / state_action_tree[state][edge]['N']
                        + c * np.sqrt(np.log(state_action_tree[state]['N']) / state_action_tree[state][edge]['N'])
                    )
                    edge_values.append((edge, state_action, UCB))
                selected_edge = max(edge_values, key=lambda x: x[2])[0]
                traversal.append((state, selected_edge))
                selected_node = c4.drop_piece(selected_node, temp_piece, selected_edge)
                temp_piece = temp_piece.get_opponent_piece()

            if expand_node is None:
                continue # terminal, backpropped already, continue to next iteration

            # simulate
            while True:
                available_edges = c4.get_available_cols(expand_node)
                state = c4.make_board_hashable(expand_node)
                selected_edge = random.choice(available_edges)
                expand_node = c4.drop_piece(expand_node, temp_piece, selected_edge)
                temp_piece = temp_piece.get_opponent_piece()

                if c4.is_terminal(expand_node) and len(traversal) > 0:
                    backprop(expand_node, piece, traversal, state_action_tree)
                    break
            
        root_dict = state_action_tree[c4.make_board_hashable(root_node)]
        if xray == True:
            print(f"After {n_iterations} simulations, the state_action_tree for root_node looks like this:")
            pprint(root_dict, indent=4)

        # now that we've updated state_action_tree via a bunch of simulations, we want to pick the edge
        # from our root node that has the highest visit count
        root_edges = []
        for action, values in root_dict.items():
            if action == 'N':
                continue
            root_edges.append((action, values['N']))
        return max(root_edges, key=lambda x: x[1])[0]
                    
    return mcts_player