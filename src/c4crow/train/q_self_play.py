import os
import time
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F

import c4crow.c4_engine as c4
from c4crow.players.q_player import QPlayer

# Set random seeds
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

WIN = 1
DRAW = 0
LOSE = -1
TIME_COST = -0.05
UPDATE_WR_THRESHOLD = 0.9

# Keep playing against self until we hit 90% win rate
# Then update opponent with own weights

# TODOS:
# 1. Optimize at every step instead of every game.
# 2. Update lagged_model every lag_interval steps.
# 3. Feels increasingly more like we may need to have separate training play and play for evaluation/visualization.
# 4. Add compact board display to c4_engine. Takes a list of boards and prints them compact on same line.
# example:
# P                   O                   P                   O                   P                   O
# | | | | | | | |     | | | | | | | |     | | | | | | | |     | | | | | | | |     | | | | | | | |     | | | | | | | |
# | | | | | | | |     | | | | | | | |     | | | | | | | |     | | | | | | | |     | | | | | | | |     | | | | | | | |
# | | | | | | | |     | | | | | | | |     | | | | | | | |     | | | | | | | |     | | | | | | | |     | | | | | | | |
# | | | | | | | |     | | | | | | | |     | | | | | | | |     | | | | | | | |     | | | | | | | |     | | | | | | | |
# | | | | | | | |     | | | |X| | | |     | | | |X| | | |     | | | |X| | | |     | | | |X| | | |     | | | |X| | | |
# | | | |O| | | |     | | | |O| | | |     | | | |O|O| | |     | | |X|O|O| | |     | | |X|O|O|O| |     | | |X|O|O|O|X|

model_arch = "SimpleConvDQN"
target_player = QPlayer(model_arch=model_arch, path_to_weights=None)
target_model = target_player.model
lagged_model = target_player.model
opponent_player = QPlayer(model_arch="SimpleConvDQN", path_to_weights=None)
device = target_player.device

def play_one_game(show_compact=False):
    # There's no need to care about piece when we save game_memory since we convert it to double channel
    # and the first channel will always be the target player
    board = c4.create_board()
    n_moves = 0
    boards = [board.copy()] if show_compact else []

    # Random coin flip to determine who goes first
    target_piece = random.choice([c4.P1, c4.P2])
    current_piece = c4.P1  # Game always starts with P1

    while True:
        if current_piece == target_piece:
            n_moves += 1
            col_idx = target_player.make_move(board, current_piece)
        else:
            col_idx = opponent_player.make_move(board, current_piece)

        board = c4.drop_piece(board, current_piece, col_idx)
        if show_compact: boards.append(board.copy())
        game_status = c4.check_win(board, current_piece)

        if game_status != "not done":
            if game_status == "win":
                result = "win" if current_piece == target_piece else "loss"
            else:
                result = "draw"
            if show_compact:
                n_boards_per_row = 6
                n_rows = (len(boards) + n_boards_per_row - 1) // n_boards_per_row
                for i in range(n_rows):
                    c4.display_compact_boards(boards[i*n_boards_per_row:(i+1)*n_boards_per_row])
            return [result, n_moves]

        current_piece = c4.get_opponent_piece(current_piece)

def evaluate(n_games=1000):
    wins = draws = total_moves_win = total_moves_draw = 0

    for _ in range(n_games):
        result, moves = play_one_game()
        if result == "win":
            wins += 1
            total_moves_win += moves
        elif result == "draw":
            draws += 1
            total_moves_draw += moves

    win_rate = wins / n_games
    draw_rate = draws / n_games
    avg_moves_to_win = total_moves_win / wins if wins else 0
    avg_moves_to_draw = total_moves_draw / draws if draws else 0

    return win_rate, draw_rate, avg_moves_to_win, avg_moves_to_draw

def optimize(optimizer, game_memory, batch_size=64, reward_decay=0.99):
    if len(game_memory) < batch_size:
        return 0

    transitions = random.sample(game_memory, batch_size)
    batch = list(zip(*transitions)) # convert list of row sublists into 4 column sublists

    state_batch = torch.tensor(np.stack(batch[0], axis=0), dtype=torch.float, device=device) # Bx2x6x7
    action_batch = torch.tensor(batch[1], dtype=torch.long, device=device).unsqueeze(1) # Bx1
    reward_batch = torch.tensor(batch[2], dtype=torch.float, device=device) # B
    
    next_states = batch[3]
    next_state_mask = torch.tensor([s is not None for s in next_states], dtype=torch.long, device=device)
    next_state_batch = torch.tensor(np.stack([s if s is not None else np.zeros_like(batch[0][0]) for s in next_states], axis=0), dtype=torch.float32, device=device) # Bx2x6x7

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = target_model(state_batch).gather(1, action_batch)

    # Compute Q(s_{t+1}) for all next states - it takes the max action value
    # we use max(1)[0] since pytorch max returns (max, max_indices) and we only want the max
    # we detach the tensor to prevent backpropogation through the lagged model
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[next_state_mask] = lagged_model(next_state_batch).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * reward_decay) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1)) # both are Bx1

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def learning_rate_schedule(n_steps):
    return 0.02

def train(n_games, eval_interval=100, lag_interval=10):
    total_steps_done = 0
    eps_steps_done = 0
    batch_size = 512
    reward_decay = 0.999
    optimizer = torch.optim.Adam(target_model.parameters(), lr=learning_rate_schedule(0))
    timestamp = datetime.now().strftime("%Y-%m-%d_%I:%M%p")
    save_path = os.path.expanduser(f"~/rl/c4crow/rl_checkpoints/{model_arch}/{timestamp}/")
    os.makedirs(save_path, exist_ok=True)

    def update_opponent():
        nonlocal eps_steps_done
        opponent_player.model.load_state_dict(target_model.state_dict())
        eps_steps_done = 0
        play_one_game(show_compact=True)
        save_model()

    def save_model():
        save_file = os.path.join(save_path, f"model_{total_steps_done}.pth")
        torch.save(target_model.state_dict(), save_file); print(f"Model saved to {save_file}")

    game_memory = []
    total_loss = 0
    t0 = time.time()
    et0 = t0
    last_eval_steps = 0

    for game in range(n_games):

        # PLAY TRAINING GAME
        game_steps = 0
        target_piece = random.choice([c4.P1, c4.P2])
        opponent_piece = c4.get_opponent_piece(target_piece)
        board = c4.create_board()

        if opponent_piece == c4.P1: # let opponent make move, then each step is target and opponent
            opponent_action = opponent_player.make_move(board, opponent_piece)
            board = c4.drop_piece(board, opponent_piece, opponent_action)

        while True:
            optimizer.param_groups[0]['lr'] = learning_rate_schedule(total_steps_done)
            game_steps += 1

            # Target player's turn
            state = c4.double_channel_one_hot_board(board, target_piece)
            action = target_player.make_move(board, target_piece, training=True, steps_done=eps_steps_done + game_steps)
            board = c4.drop_piece(board, target_piece, action)
            game_status = c4.check_win(board, target_piece)
            if game_status != "not done":
                game_memory.append([state, action, WIN if game_status == "win" else DRAW, None])
                break
            
            # Opponent's turn
            opponent_action = opponent_player.make_move(board, opponent_piece)
            board = c4.drop_piece(board, opponent_piece, opponent_action)
            game_status = c4.check_win(board, opponent_piece)
            if game_status != "not done":
                game_memory.append([state, action, LOSE if game_status == "win" else DRAW, None])
                break
            
            next_state = c4.double_channel_one_hot_board(board, target_piece)
            game_memory.append([state, action, TIME_COST, next_state])

            if (total_steps_done + game_steps) % lag_interval == 0:
                lagged_model.load_state_dict(target_model.state_dict())
            loss = optimize(optimizer, game_memory, batch_size=batch_size, reward_decay=reward_decay)
            total_loss += loss

        # Final optimization step
        loss = optimize(optimizer, game_memory, batch_size=batch_size, reward_decay=reward_decay)
        total_loss += loss
        total_steps_done += game_steps
        eps_steps_done += game_steps

        # if len(global_game_memory) > 10000:
        #     global_game_memory = global_game_memory[-10000:]

        if game % eval_interval == 0:
            win_rate, draw_rate, avg_moves_win, avg_moves_draw = evaluate()
            edt = time.time() - et0
            eval_steps = total_steps_done - last_eval_steps
            steps_per_second = eval_steps / edt if edt > 0 else 0
            avg_loss = total_loss / eval_steps if eval_steps > 0 else 0

            print(f"G: {game}/{n_games} | S: {total_steps_done:,} | LR: {optimizer.param_groups[0]['lr']:.6f} | WR: {win_rate:.2%} | DR: {draw_rate:.2%} | "
                  f"AvgWM: {avg_moves_win:.2f} | AvgDM: {avg_moves_draw:.2f} | steps/s: {steps_per_second:.2f} | AvgL: {avg_loss:.6f}")

            et0 = time.time()
            last_eval_steps = total_steps_done
            total_loss = 0

            if win_rate > UPDATE_WR_THRESHOLD:
                update_opponent()
                print(f"\nOpponent updated at Game {game:,}, Steps {total_steps_done:,}")
    save_model()

if __name__ == "__main__":
    train(n_games=100000, eval_interval=100)
