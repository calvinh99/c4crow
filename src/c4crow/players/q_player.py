from .base_player import Player
import os
import math
import random
import numpy as np
import torch
from c4crow.models.DQN import SimpleConvDQN
import c4crow.c4_engine as c4

class QPlayer(Player):
    def __init__(self, model_arch: str, path_to_weights: str=None, eps_start=0.9, eps_end=0.05, eps_steps=2000):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if model_arch == "SimpleConvDQN":
            self.model = SimpleConvDQN().to(self.device)
        if path_to_weights and os.path.exists(path_to_weights):
            self.model.load_state_dict(torch.load(path_to_weights, map_location=self.device))
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps

    def make_move(self, board: np.ndarray, piece: int, training=False, steps_done=0) -> int:
        available_cols = c4.get_available_cols(board)
        board = c4.double_channel_one_hot_board(board, piece) # 2x6x7
        input_state = torch.tensor(board, dtype=torch.float32, device=self.device).unsqueeze(dim=0) # batch size of 1, 1x2x6x7
        
        if training:
            # e.g. with eps_start=0.9, eps_end=0.05, eps_steps=2000, the diff is 0.85
            # At step 1, exp(-1/2000) is near 1, so eps_threshold will be approx 0.05 + 0.85
            # At step 2000, exp(-2000/2000) is approx 0.367, so eps_threshold will be approx 0.05 + 0.31
            # At step 8000, exp(-8000/2000) is approx 0.018, so eps_threshold will be approx 0.05 + 0.01
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * steps_done / self.eps_steps)
        else:
            eps_threshold = 0 # always exploit

        if random.random() > eps_threshold: # the higher epsilon, the more likely we choose random (exploration)
            with torch.no_grad():
                state_action_values = self.model(input_state)[0].cpu().numpy()
                state_action_values = [state_action_values[col] for col in available_cols]
                return available_cols[np.argmax(state_action_values)]
        else:
            return random.choice(available_cols)
    
    def move_to_device(self, device: torch.device):
        self.device = device
        self.model.to(device)

    def make_batch_move(self, boards: np.ndarray, piece: np.ndarray, training=False) -> np.ndarray:
        pass