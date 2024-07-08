import yaml
from inspect import getfullargspec
import random
from enum import Enum
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F

from c4crow.players import get_rl_player, random_player
import c4crow.c4_engine as c4

class Rewards(Enum):
    TIME_COST = -0.05
    LOSE = -1
    WIN = 1
    DRAW = 0.5

def save_config(save_dir, config):
    config_path = os.path.join(save_dir, "config.yml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to {config_path}")

def get_function_info(func):
    return {
        'name': func.__name__,
        'module': func.__module__,
        'args': getfullargspec(func).args
    }

def optimizer_to_dict(optimizer):
    optimizer_dict = {
        'name': optimizer.__class__.__name__,
        'params': {}
    }
    for group in optimizer.param_groups:
        for key, value in group.items():
            if key != 'params':  # 'params' contains the actual parameters, which we don't need to save
                optimizer_dict['params'][key] = value
    return optimizer_dict

def save_checkpoint(save_dir, policy_net, game_number):
    checkpoint_path = os.path.join(save_dir, f"model_{game_number}.pth")
    torch.save(policy_net.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def save_eval_history(save_dir, eval_history, game_number):
    eval_history_path = os.path.join(save_dir, f"eval_history_{game_number}.csv")
    pd.DataFrame(eval_history, columns=['game', 'win_rate', 'avg_moves_to_win']).to_csv(eval_history_path, index=False)
    print(f"Evaluation history saved to {eval_history_path}")

def save_best_checkpoint(save_dir, policy_net, best_win_rate, game_number):
    best_checkpoint_path = os.path.join(save_dir, "best_model.pth")
    torch.save(policy_net.state_dict(), best_checkpoint_path)
    print(f"New best model (win rate: {best_win_rate:.4f}) saved at game {game_number}")

def estimate_model_size(model):
    n_params = sum(p.numel() for p in model.parameters())
    memory_bits = sum((p.numel() * p.element_size() * 8 * 2) for p in model.parameters())
    memory_bits += sum((b.numel() * b.element_size() * 8) for b in model.buffers())
    memory_mb = memory_bits / (8 * 1024 * 1024)  # Convert bits to MB
    return n_params, memory_mb

def eval_winrate(train_player, against_player, n_games=100):
    # returns winrate and avg moves taken for wins
    moves_to_win = []
    for i in range(n_games):
        board = c4.create_board()
        moves_taken = 0

        while True:
            col_idx = train_player(board, c4.Pieces.P1) # greedy, training=False
            board = c4.drop_piece(board, c4.Pieces.P1, col_idx)
            moves_taken += 1

            if c4.check_win(board, c4.Pieces.P1):
                moves_to_win.append(moves_taken)
                break
            elif len(c4.get_available_cols(board)) == 0:
                break

            col_idx = against_player(board, c4.Pieces.P2) # greedy, training=False
            board = c4.drop_piece(board, c4.Pieces.P2, col_idx)
            if c4.check_win(board, c4.Pieces.P2) or len(c4.get_available_cols(board)) == 0:
                break
    return len(moves_to_win)/n_games, sum(moves_to_win)/len(moves_to_win)

def plot_eval_history(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Plot win rate on the first y-axis
    ax1.plot(df['Game'], df['Win Rate'], 'b-', label='Win Rate')
    ax1.set_xlabel('Number of Games')
    ax1.set_ylabel('Win Rate', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Plot average moves to win on the second y-axis
    ax2.plot(df['Game'], df['Avg Moves to Win'], 'r-', label='Avg Moves to Win')
    ax2.set_ylabel('Avg Moves to Win', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Add title and legend
    plt.title('Training Progress')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # Show the plot
    plt.tight_layout()
    plt.show()

def optimize_model(optimizer, policy_net, target_net, batch_size, reward_decay, rl_memory, device):
    if len(rl_memory) < batch_size:
        return
    transitions = np.array(random.sample(rl_memory, batch_size), dtype=object) # (batch_size, 4)
    # Use array indexing to separate columns
    state0_batch = np.stack(transitions[:, 0]) # (batch_size, N_ROWS, N_COLS)
    action_batch = transitions[:, 1].astype(np.float32) # (batch_size, )
    reward_batch = transitions[:, 2].astype(np.float32) # (batch_size, )

    # handle next state batch
    state1_list = transitions[:, 3]
    state1_mask = np.array([s is not None for s in state1_list])
    state1_batch = np.array([s if s is not None else np.zeros_like(state0_batch[0]) for s in state1_list])  # (batch_size, N_ROWS, N_COLS)

    # convert to tensors, float despite int values since our nets use float
    state0_batch = torch.tensor(state0_batch, dtype=torch.float, device=device)
    action_batch = torch.tensor(action_batch, dtype=torch.long, device=device) # expect int64 or long for index
    reward_batch = torch.tensor(reward_batch, dtype=torch.float, device=device)
    state1_mask = torch.tensor(state1_mask, dtype=torch.long, device=device)
    state1_batch = torch.tensor(state1_batch, dtype=torch.float, device=device)

    # reshape state batches for CNN
    state0_batch = state0_batch.unsqueeze(1) # (batch_size, 1, N_ROWS, N_COLS) extra 1 is for channel dimension since CNNs expected (n_channels, n_rows, n_cols)
    state1_batch = state1_batch.unsqueeze(1)
    action_batch = action_batch.unsqueeze(1) # make it shape (batch_size, 1) for indexing, needs 2 dims since policy_net output is (batch_size, N_COLS) which is 2 dims

    # prediction from policy net
    # SARSA, take only the prob of the action actually taken
    state_action_values = policy_net(state0_batch).gather(1, action_batch) # (batch_size, 1)

    # Output is [batch_size, N_COLS] -> we max across N_COLS and get (batch_size,)
    # we use max(1)[0] since max(1) will return the maxed tensor and the max indices and we only want the tensor
    next_state_values = torch.zeros(batch_size, device=device) # for timesteps where state1 is end of game, these values will be 0
    next_state_values[state1_mask] = target_net(state1_batch).max(1)[0].detach() # idk: why detach? (batch_size, )
    expected_state_action_values = (next_state_values * reward_decay) + reward_batch # (batch_size, )
    expected_state_action_values = expected_state_action_values.unsqueeze(1) # (batch_size, 1)

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values) # e.g. tensor(0.0568, grad_fn=<SmoothL1LossBackward0>)

    optimizer.zero_grad() # so that previous gradients dont affect this batch
    loss.backward() # backprop, chain rule
    optimizer.step() # subtract weights by grad

def train(
    path_to_weights: str,
    architecture: str,
    against_player,
    reward_decay=0.999, # high emphasis on future rewards
    n_games=20000,
    batch_size=96,
    learning_rate=1e-3,
    EVAL_INTERVAL=20,
    PRINT_EVAL_INTERVAL=100,
    TARGET_UPDATE_INTERVAL=10, # update lagged target net every 10 games
    CHECKPOINT_SAVE_INTERVAL=1000
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rl_memory = [] # list of lists of [state0, action_col_idx, state1, reward]

    # I need to make this so that we have games where RL player goes second as well
    train_player, policy_net = get_rl_player(path_to_weights, architecture)
    _, target_net = get_rl_player(path_to_weights, architecture) # lagged copy of policy net
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    steps_done = 0
    eval_history = []

    # create save dir name based on current datetime and add it to policy_net, e.g. policy_net_2024-07-05_00-00-00
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"rl_checkpoints/{architecture}_policy_net_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    n_params, mem_reqs = estimate_model_size(policy_net)

    # Prepare configuration
    config = {
        'path_to_weights': path_to_weights,
        'architecture': architecture,
        'against_player': get_function_info(against_player),
        'reward_decay': reward_decay,
        'n_games': n_games,
        'batch_size': batch_size,
        'EVAL_INTERVAL': EVAL_INTERVAL,
        'device': str(device),
        'timestamp': timestamp,
        'optimizer': optimizer_to_dict(optimizer),
        'trainable_parameters': n_params,
        'estimated_RAM_usage_MB': f"{mem_reqs:.4f}",
        'save_directory': save_dir,
        'PRINT_EVAL_INTERVAL': PRINT_EVAL_INTERVAL,
        'TARGET_UPDATE_INTERVAL': TARGET_UPDATE_INTERVAL,
        'CHECKPOINT_SAVE_INTERVAL': CHECKPOINT_SAVE_INTERVAL,
    }

    # Save configuration
    save_config(save_dir, config)

    # Print configuration
    print("Training start.")
    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))

    best_win_rate = 0.0

    # start playing games non stop, for N_TURNS
    for i in range(n_games):
        if (i % EVAL_INTERVAL) == (EVAL_INTERVAL - 1):
            win_rate, avg_moves_to_win = eval_winrate(train_player, against_player)
            eval_history.append([i+1, win_rate, avg_moves_to_win])

            # Save best checkpoint after 1000 games
            if i >= 1000 and win_rate > best_win_rate:
                best_win_rate = win_rate
                save_best_checkpoint(save_dir, policy_net, best_win_rate, i+1)

        # Save regular checkpoint and evaluation history
        if (i % CHECKPOINT_SAVE_INTERVAL) == (CHECKPOINT_SAVE_INTERVAL - 1):
            save_checkpoint(save_dir, policy_net, i+1)
            save_eval_history(save_dir, eval_history, i+1)

        if (i % PRINT_EVAL_INTERVAL) == (PRINT_EVAL_INTERVAL - 1):
            print(f'Game {i+1}: | win_rate: {eval_history[-1][1]:.4f} | moves_taken: {eval_history[-1][2]:.2f}')

        p1_state0 = c4.create_board()

        while True:
            steps_done += 1

            # we only record target player actions for our training
            # our model will train to treat itself as player 1 always
            p1_col_idx = train_player(p1_state0, c4.Pieces.P1, training=True, steps_done=steps_done)
            p1_state1 = c4.drop_piece(p1_state0, c4.Pieces.P1, p1_col_idx)
            if c4.check_win(p1_state1, c4.Pieces.P1):
                rl_memory.append([p1_state0, p1_col_idx, Rewards.WIN.value, None])
                break
            elif len(c4.get_available_cols(p1_state1)) == 0:
                rl_memory.append([p1_state0, p1_col_idx, Rewards.DRAW.value, None])
                break

            # penalize for losses
            p2_state0 = p1_state1
            p2_col_idx = against_player(p2_state0, c4.Pieces.P2) # to train against itself we would need to flip board states so 1 -> 2, 2 -> 1, and then flip back.
            p2_state1 = c4.drop_piece(p2_state0, c4.Pieces.P2, p2_col_idx)
            if c4.check_win(p2_state1, c4.Pieces.P2):
                rl_memory.append([p1_state0, p1_col_idx, Rewards.LOSE.value, None])
                break
            elif len(c4.get_available_cols(p2_state1)) == 0:
                rl_memory.append([p1_state0, p1_col_idx, Rewards.DRAW.value, None])
                break

            # penalize for time
            rl_memory.append([p1_state0, p1_col_idx, Rewards.TIME_COST.value, p2_state1]) # state before any move, then state after p1 and p2 makes a move
            p1_state0 = p2_state1

            # lagged copy of target net every some steps, so after update, if interval is 10, the lag btwn target and policy becomes 1, 2, 3, ... 8, 9, 0
            if i % TARGET_UPDATE_INTERVAL == TARGET_UPDATE_INTERVAL - 1:
                target_net.load_state_dict(policy_net.state_dict())

            # optimize model
            optimize_model(optimizer, policy_net, target_net, batch_size, reward_decay, rl_memory, device)

    print("Finished training.")
    save_checkpoint(save_dir, policy_net, n_games)
    save_eval_history(save_dir, eval_history, n_games)
    # plot_eval_history(eval_history_path)

if __name__ == "__main__":
    # train against random
    train(None, "DQN2", random_player)