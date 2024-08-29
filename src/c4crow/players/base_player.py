from abc import ABC, abstractmethod
import numpy as np

class Player(ABC):
    @abstractmethod
    def make_move(self, board: np.ndarray, piece: int) -> int:
        pass