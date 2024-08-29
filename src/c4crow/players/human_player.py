from .base_player import Player
import numpy as np
import c4crow.c4_engine as c4

class HumanPlayer(Player):
    def make_move(self, board: np.ndarray, piece: int) -> int:
        while True:
            try:
                col = int(input(f"Player {piece}, enter your move (0-6): "))
                if col in c4.get_available_cols(board):
                    return col
                else:
                    print("Invalid move. Please choose an available column.")
            except ValueError:
                print("Invalid input. Please enter a number between 0 and 6.")
