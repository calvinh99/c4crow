from .base_player import Player
import numpy as np
import random
import c4crow.c4_engine as c4

class RandomPlayer(Player):
    def make_move(self, board: np.ndarray, piece: int) -> int:
        available_cols = c4.get_available_cols(board)
        return random.choice(available_cols)