import time
import argparse
from typing import Tuple
import numpy as np

import c4crow.c4_engine as c4
from c4crow.players import Player,HumanPlayer, RandomPlayer, QPlayer, MinimaxPlayer, MCTSPlayer

PRETTY_LINE = "\n" + "█" * 100 + "\n"

def play_game(player1: Player, player2: Player, ux_time: int = 0) -> Tuple[np.ndarray, int]:
    board = c4.create_board()
    current_piece = c4.P1
    piece_to_player = {c4.P1: player1, c4.P2: player2}

    print(PRETTY_LINE)
    c4.display_board(board)
    while True:
        print(f"Player {current_piece}'s turn. Player is thinking...\n")
        col_idx = piece_to_player[current_piece].make_move(board, current_piece)
        board = c4.drop_piece(board, current_piece, col_idx)

        print(f"Player {current_piece} placed a piece in column {col_idx}.\n")
        c4.display_board(board)
        print(PRETTY_LINE)
                
        game_status = c4.check_win(board, current_piece)
        if game_status == "win":
            return board, current_piece
        elif game_status == "draw":
            return board, None
        
        time.sleep(ux_time)

        current_piece = c4.get_opponent_piece(current_piece)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Connect4 with different players.")
    parser.add_argument('--player1', choices=['human', 'dqn', 'minimax', 'mcts', 'random'], default='human', help='Player 1 type')
    parser.add_argument('--player2', choices=['human', 'dqn', 'minimax', 'mcts', 'random'], default='human', help='Player 2 type')
    parser.add_argument('--wait', type=int, default=0, help='Wait time between moves (seconds)')
    args = parser.parse_args()

    print("Welcome to Connect4!")

    def get_player(player_type):
        if player_type == 'human':
            return HumanPlayer()
        elif player_type == 'dqn':
            return QPlayer("SimpleConvDQN", "/home/calvinhuang/rl/c4crow/rl_checkpoints/SimpleConvDQN/2024-09-01_12:59AM/model_741359.pth")
        elif player_type == 'minimax':
            return MinimaxPlayer(max_depth=4, xray=True)
        elif player_type == 'mcts':
            return MCTSPlayer(n_iterations=10000, xray=True)
        elif player_type == 'random':
            return RandomPlayer()

    player1 = get_player(args.player1)
    player2 = get_player(args.player2)

    final_board, winning_piece = play_game(player1, player2, args.wait)
    
    if winning_piece:
        print(f"Player {winning_piece} wins!")
    else:
        print("It's a draw!")