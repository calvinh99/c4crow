import multiprocessing
from multiprocessing import Value, Manager, Lock
import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import logging
from datetime import datetime
import os
from copy import deepcopy
import time

import c4crow.c4_engine as c4
from c4crow.players.q_player import QPlayer

multiprocessing.set_start_method('spawn', force=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
UPDATE_WR_THRESHOLD = 0.55

def play_one_game(target_player: QPlayer, opponent_player: QPlayer, show_compact=False) -> Tuple[str, int]:
    board = c4.create_board()
    n_moves = 0
    boards = [board.copy()] if show_compact else []
    target_piece = random.choice([c4.P1, c4.P2])
    current_piece = c4.P1

    if show_compact: print(f"Target piece: {target_piece}")

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
            return result, n_moves

        current_piece = c4.get_opponent_piece(current_piece)

def evaluate(current_player: QPlayer, best_player: QPlayer, n_games: int = 800) -> Tuple[float, float, float, float]:
    wins = draws = total_moves_win = total_moves_draw = 0
    for _ in range(n_games):
        result, moves = play_one_game(current_player, best_player)
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

def save_model_weights(model, save_path):
    torch.save(model.state_dict(), save_path); print(f"Model saved to {save_path}")

def sample_from_buffer(shared_game_memory: List[List[Any]], batch_size: int) -> List[List[Any]]:
    buffer_size = len(shared_game_memory)
    if buffer_size < batch_size:
        return []
    ix = np.random.choice(buffer_size, size=batch_size, replace=False)
    return [shared_game_memory[i] for i in ix]

def optimize(target_model: torch.nn.Module, lagged_model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
             batch: List[List[Any]], device: torch.device, batch_size: int, reward_decay: float) -> float:
    batch = list(zip(*batch))

    state_batch = torch.tensor(np.stack(batch[0], axis=0), dtype=torch.float, device=device)
    action_batch = torch.tensor(batch[1], dtype=torch.long, device=device).unsqueeze(1)
    reward_batch = torch.tensor(batch[2], dtype=torch.float, device=device)
    
    next_states = batch[3]
    next_state_mask = torch.tensor([s is not None for s in next_states], dtype=torch.long, device=device)
    next_state_batch = torch.tensor(np.stack([s if s is not None else np.zeros_like(batch[0][0]) for s in next_states], axis=0), dtype=torch.float32, device=device)

    state_action_values = target_model(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[next_state_mask] = lagged_model(next_state_batch).max(1)[0].detach()

    expected_state_action_values = (next_state_values * reward_decay) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def self_play_process(data_lock: Lock, shared_game_memory: List[List[Any]], shared_models: Dict[str, QPlayer], shared_counters: Dict[str, Value]) -> None:
    logger = logging.getLogger('SelfPlay')
    best_player = shared_models['best_player']
    best_player.move_to_device(torch.device('cuda:0'))
    games_played = 0
    t0 = time.time()
    pt = t0
    last_num_steps = shared_counters['total_steps'].value
    log_interval = 1000
    eps_steps_reset_threshold = 4000
    
    while True:
        games_played += 1
        eps_steps_done = shared_counters['eps_steps'].value
        
        # Step 1. Play Training Game
        game_memory: List[List[Any]] = []
        game_steps = 0
        target_piece = random.choice([c4.P1, c4.P2])
        opponent_piece = c4.get_opponent_piece(target_piece)
        board = c4.create_board()

        if opponent_piece == c4.P1:
            opponent_action = best_player.make_move(board, opponent_piece)
            board = c4.drop_piece(board, opponent_piece, opponent_action)

        while True:
            game_steps += 1

            # Target player's turn
            state = c4.double_channel_one_hot_board(board, target_piece)
            action = best_player.make_move(board, target_piece, training=True, steps_done=eps_steps_done + game_steps)
            board = c4.drop_piece(board, target_piece, action)
            game_status = c4.check_win(board, target_piece)
            if game_status != "not done":
                game_memory.append([state, action, WIN if game_status == "win" else DRAW, None])
                break
            
            # Opponent's turn
            opponent_action = best_player.make_move(board, opponent_piece)
            board = c4.drop_piece(board, opponent_piece, opponent_action)
            game_status = c4.check_win(board, opponent_piece)
            if game_status != "not done":
                game_memory.append([state, action, LOSE if game_status == "win" else DRAW, None])
                break

            next_state = c4.double_channel_one_hot_board(board, target_piece)
            game_memory.append([state, action, TIME_COST, next_state])
        
        # with data_lock:
        shared_game_memory.extend(game_memory)

        # dont care about thread safety for these counters
        shared_counters['total_steps'].value += game_steps
        shared_counters['eps_steps'].value += game_steps
        if shared_counters['eps_steps'].value > eps_steps_reset_threshold:
            shared_counters['eps_steps'].value = 0

        # Logging
        if games_played % log_interval == 0:
            ct = time.time()
            dt = ct - pt
            steps_per_second = (shared_counters['total_steps'].value - last_num_steps) / dt
            logger.info(f"[SELF_PLAY] Games: {games_played} | Total Steps: {shared_counters['total_steps'].value} | "
                        f"Eps Steps: {shared_counters['eps_steps'].value} | "
                        f"Best Player Checkpoint: {shared_models['best_player_checkpoint_path']} | "
                        f"{steps_per_second:.2f} steps/sec")
            pt = time.time()
            last_num_steps = shared_counters['total_steps'].value

def optimize_process(data_lock: Lock, shared_game_memory: List[List[Any]], shared_models: Dict[str, QPlayer], shared_counters: Dict[str, Value]) -> None:
    logger = logging.getLogger('Optimize')
    current_player = QPlayer(model_arch="SimpleConvDQN")
    current_player.model.load_state_dict(shared_models['best_player'].model.state_dict())
    current_player.move_to_device(torch.device('cuda:0'))
    lagged_model = deepcopy(current_player.model)

    total_loss = 0
    batch_size = 2048
    lr = 0.002
    reward_decay = 0.999
    eval_interval = 1000
    optimizer = torch.optim.AdamW(current_player.model.parameters(), lr=lr)
    t0 = time.time()
    pt = t0
    sample_time = 0
    log_interval = 100
    
    while True:
        # Sample from shared buffer and optimize
        sample_t0 = time.time()
        batch = sample_from_buffer(shared_game_memory, batch_size)
        sample_time += time.time() - sample_t0
        if len(batch) < batch_size:
            continue
        loss = optimize(current_player.model, lagged_model, optimizer, batch, current_player.device, batch_size, reward_decay)
        total_loss += loss
        lagged_model.load_state_dict(current_player.model.state_dict())
        
        # Evaluate and update best player
        # with data_lock:
        # Not using lock. Don't care about thread safety for this since it's just a counter value, can't see it going bad.
        # When using lock ~1 or less opt step == 1 game
        shared_counters['opt_steps'].value += 1

        if shared_counters['opt_steps'].value % log_interval == 0:
            ct = time.time()
            dt = ct - pt
            steps_per_second = log_interval / dt
            avg_loss = total_loss / log_interval / batch_size
            avg_sample_time = sample_time / log_interval
            logger.info(f"[OPTIMIZE] Opt Steps: {shared_counters['opt_steps'].value} | "
                        f"{steps_per_second:.2f} steps/sec| Avg Sample Time: {avg_sample_time:.2f}s | "
                        f"Avg Loss: {avg_loss:.6f}")
            pt = time.time()
            sample_time = 0

        if shared_counters['opt_steps'].value % eval_interval == 0:
            win_rate, draw_rate, avg_moves_win, avg_moves_draw = evaluate(current_player, shared_models['best_player'])
            logger.info(f"[EVALUATION] | "
                        f"Win Rate: {win_rate:.2%} | Draw Rate: {draw_rate:.2%} | "
                        f"Avg Moves to Win: {avg_moves_win:.2f}| Avg Moves to Draw: {avg_moves_draw:.2f}")
            
            if win_rate > UPDATE_WR_THRESHOLD:
                logger.info("Updating best player...")
                logger.info("Playing one game between current player and best player...")
                play_one_game(current_player, shared_models['best_player'], show_compact=True)
                with data_lock:
                    shared_models['best_player'].model.load_state_dict(current_player.model.state_dict())
                    save_path = os.path.join(shared_models['save_path'], f"checkpoint_{shared_counters['opt_steps'].value}.pth")
                    save_model_weights(current_player.model, save_path)
                    shared_models['best_player_checkpoint_path'] = save_path
                logger.info(f"[OPTIMIZE] Updated best player at Opt Steps: {shared_counters['opt_steps'].value}")
            
            total_loss = 0

def train() -> None:
    manager = Manager()
    data_lock = manager.Lock()
    shared_game_memory: List[List[Any]] = manager.list()
    shared_models: Dict[str, QPlayer] = manager.dict()
    shared_counters: Dict[str, Value] = {
        'total_steps': manager.Value('i', 0),
        'eps_steps': manager.Value('i', 0),
        'opt_steps': manager.Value('i', 0)
    }
    
    # Initialize best player
    best_player = QPlayer(model_arch="SimpleConvDQN")
    best_player.move_to_device(torch.device('cpu'))
    shared_models['best_player'] = best_player
    shared_models['best_player_checkpoint_path'] = "Random init"
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%I:%M%p")
    save_path = os.path.expanduser(f"~/rl/c4crow/rl_checkpoints/SimpleConvDQN/{timestamp}/")
    os.makedirs(save_path, exist_ok=True)
    shared_models['save_path'] = save_path
    logging.info(f"Saving checkpoints to: {save_path}")
    
    # Start processes
    self_play_proc = multiprocessing.Process(target=self_play_process, args=(data_lock, shared_game_memory, shared_models, shared_counters))
    optimize_proc = multiprocessing.Process(target=optimize_process, args=(data_lock, shared_game_memory, shared_models, shared_counters))
    
    self_play_proc.start()
    optimize_proc.start()
    
    try:
        self_play_proc.join()
        optimize_proc.join()
    except KeyboardInterrupt:
        logging.info("Training interrupted. Saving final model...")
        torch.save(shared_models['best_player'].model.state_dict(), os.path.join(save_path, f"final_model.pth"))
        logging.info("Final model saved. Exiting...")

if __name__ == "__main__":
    train()
