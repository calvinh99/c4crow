import time
from typing import Callable, Tuple
import c4crow.c4_engine as c4
from c4crow.players import real_player, random_player, get_rl_player, get_minimax_player

PRETTY_LINE = "\n" + "█" * 100 + "\n"

def play_game(player1: Callable, player2: Callable, ux_time: int = 0) -> Tuple[c4.np.ndarray, c4.Pieces]:
    board = c4.create_board()
    current_piece = c4.Pieces.P1
    piece_to_player = {c4.Pieces.P1: player1, c4.Pieces.P2: player2}

    print(PRETTY_LINE)
    c4.display_board(board)
    while True:
        # let player make a move
        print(f"Player {current_piece.display_str}'s turn. Player is thinking...\n")
        col_idx = piece_to_player[current_piece](board, current_piece)
        board = c4.drop_piece(board, current_piece, col_idx)

        print(f"Player {current_piece.display_str} placed a piece in column {col_idx}.\n")
        c4.display_board(board)
        print(PRETTY_LINE)
                
        if c4.check_win(board, current_piece):
            return board, current_piece

        if len(c4.get_available_cols(board)) == 0:
            return board, None  # Draw
        
        time.sleep(ux_time) # let user register the move that just happened if it did not result in a win

        current_piece = c4.Pieces.P2 if current_piece == c4.Pieces.P1 else c4.Pieces.P1

if __name__ == "__main__":
    print("Welcome to Connect4!")
    # final_board, winning_piece = play_game(real_player, real_player)

    # TODO: another flaw, the rl model thinks it's "1". So perhaps we need something like a "flip_board_for_rl" which
    # temporarily swaps all the values so that the rl_player is "1", then swaps back only for displaying and checking 
    # wins. Got to remember to keep track of real_piece so that the rl output is placed using that piece.
    # final_board, winning_piece = play_game(get_rl_player("./model_weights_DQN_CNN.pth"), real_player)

    # final_board, winning_piece = play_game(
    #     get_rl_player("./model_weights_DQN_CNN.pth"),
    #     get_minimax_player(max_depth=4, xray=True),
    #     5
    # )

    # FOUND A BIG BUG in the Minimax, if given obvious opportunity to win it won't! Because if the future
    # score for an earlier column is a win, then it's already infinity! So if another infinity it won't update!
    # We need to add a gamma like 0.9 that multiplies to the score of each turn, so that earlier turns are prioritized!
    # And we shouldn't use infinity, we should use a large positive number, e.g. 1 million.
    # Note: the reason we minimize opponent score is not to make it make the worst moves, but rather the best moves!
    #       why min instead of also max? Simply because we need a way to differentiate for each player the outcomes
    #       from opponents. E.g. if my opponent also max then when it's my turn and I see two branches that both
    #       return 1 million score, which is my win and which is my opponent's win? If opponent minimizes then 
    #       it's clear the -1 million is my loss!
    final_board, winning_piece = play_game(
        random_player,
        get_minimax_player(max_depth=4, xray=True),
        5
    )
    
    if winning_piece:
        print(f"Player {winning_piece.display_str} wins!")
    else:
        print("It's a draw!")