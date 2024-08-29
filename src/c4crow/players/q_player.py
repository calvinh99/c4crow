from .base_player import Player
import os
import math
import random
import numpy as np
import torch
from c4crow.models.DQN import DQN, DQN2
import c4crow.c4_engine as c4

class QPlayer(Player):
    def __init__(self, path_to_weights: str, architecture: str, eps_start=0.9, eps_end=0.05, eps_steps=2000):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = DQN2(c4.N_COLS).to(self.device) if architecture == "DQN2" else DQN(c4.N_COLS).to(self.device)
        if path_to_weights and os.path.exists(path_to_weights):
            self.model.load_state_dict(torch.load(path_to_weights, map_location=self.device))
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.steps_done = 0

    def make_move(self, board: np.ndarray, piece: int, training=False) -> int:
        available_cols = c4.get_available_cols(board)
        input_state = torch.tensor(board, dtype=torch.float, device=self.device).unsqueeze(dim=0).unsqueeze(dim=0)
        
        if training:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * self.steps_done / self.eps_steps)
            self.steps_done += 1
        else:
            eps_threshold = 0

        if random.random() > eps_threshold:
            with torch.no_grad():
                state_action_values = self.model(input_state)[0].cpu().numpy()
                state_action_values = [state_action_values[col] for col in available_cols]
                return available_cols[np.argmax(state_action_values)]
        else:
            return random.choice(available_cols)