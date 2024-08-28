import time
import argparse
from typing import Callable, Tuple
import numpy as np

import c4crow.c4_engine as c4
from c4crow.players import real_player, random_player, get_dqn_player, get_minimax_player, get_mcts_player

PRETTY_LINE = "\n" + "█" * 100 + "\n"

def play_game(player1_func: Callable, player2_func: Callable, ux_time: int = 0) -> Tuple[np.ndarray, int]:
    board = c4.create_board()
    current_piece = c4.P1
    piece_to_player_func = {c4.P1: player1_func, c4.P2: player2_func}

    print(PRETTY_LINE)
    c4.display_board(board)
    while True:
        # let player make a move
        print(f"Player {current_piece}'s turn. Player is thinking...\n")
        col_idx = piece_to_player_func[current_piece](board, current_piece)
        board = c4.drop_piece(board, current_piece, col_idx)

        print(f"Player {current_piece} placed a piece in column {col_idx}.\n")
        c4.display_board(board)
        print(PRETTY_LINE)
                
        game_status = c4.check_win(board, current_piece)
        if game_status == "win":
            return board, current_piece
        elif game_status == "draw":
            return board, None
        
        time.sleep(ux_time) # let user register the move that just happened if it did not result in a win

        current_piece = c4.get_opponent_piece(current_piece)

if __name__ == "__main__":
    # python src/c4crow/play.py --player1 human --player2 dqn --wait 0
    parser = argparse.ArgumentParser(description="Play Connect4 with different players.")
    parser.add_argument('--player1', choices=['human', 'dqn', 'minimax', 'mcts', 'random'], default='human', help='Player 1 type')
    parser.add_argument('--player2', choices=['human', 'dqn', 'minimax', 'mcts', 'random'], default='human', help='Player 2 type')
    parser.add_argument('--wait', type=int, default=3, help='Wait time between moves (seconds)')
    args = parser.parse_args()

    print("Welcome to Connect4!")

    arg_to_player_func = {
        "human": real_player,
        "dqn": get_dqn_player("/home/calvinhuang/rl/c4crow/rl_checkpoints/DQN2_2024-08-26_21-12-02/model_18000.pth", "DQN2")[0],
        "minimax": get_minimax_player(max_depth=4, xray=True),
        "mcts": get_mcts_player(n_iterations=20000, xray=True),
        "random": random_player
    }

    player1_func = arg_to_player_func[args.player1]
    player2_func = arg_to_player_func[args.player2]

    final_board, winning_piece = play_game(player1_func, player2_func, args.wait)
    
    if winning_piece:
        print(f"Player {winning_piece} wins!")
    else:
        print("It's a draw!")