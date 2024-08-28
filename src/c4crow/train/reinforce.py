from collections import deque
import pandas as pd
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from c4crow.c4_engine import get_available_cols, create_board, drop_piece, Pieces, check_win
from c4crow.players import get_policy_net_player, random_player

# -------------------------
# Train
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

def eval_winrate(train_player, against_player, n_games=100):
    # returns winrate and avg moves taken for wins
    moves_to_win = []
    for i in range(n_games):
        board = create_board()
        moves_taken = 0

        while True:
            col_idx = train_player(board) # greedy, training=False
            board = drop_piece(board, Pieces.P1, col_idx)
            moves_taken += 1

            if check_win(board, Pieces.P1):
                moves_to_win.append(moves_taken)
                break
            elif len(get_available_cols(board)) == 0:
                break

            col_idx = against_player(board, Pieces.P2) # greedy, training=False
            board = drop_piece(board, Pieces.P2, col_idx)
            if check_win(board, Pieces.P2) or len(get_available_cols(board)) == 0:
                break
    return len(moves_to_win)/n_games, sum(moves_to_win)/len(moves_to_win)

def update_policy(optimizer, data, gamma=0.999):
    boards, actions, log_probs, next_boards, rewards = zip(*data)

    # each return at time step t is sum(k=0 to T-t)(gamma^(k) * r_{t+k}) where T is total number of steps
    returns = deque()
    u = 0
    for r in reversed(rewards):
        u = r + gamma * u
        returns.appendleft(u)
    returns = torch.tensor(returns).to(device)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8) # normalize

    # Calculate loss
    policy_loss = []
    for log_prob, u in zip(log_probs, returns):
        policy_loss.append(-log_prob * u) # ascent instead of descent
    policy_loss = torch.stack(policy_loss).sum()

    # update
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    return policy_loss

rewards = {
    "win": 1,
    "lose": -1,
    "draw": 0,
    "turn": 0
}
policy_piece = Pieces.P1
policy_player, policy_net = get_policy_net_player(None)
opponent_piece = policy_piece.get_opponent_piece()
opponent_player = random_player

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
print(policy_net, "\n\n\n")

n_episodes = 1000

losses = []
eval_history = []
previous_params = policy_net.fc1.weight.data.flatten().cpu().numpy().copy()

for episode in range(n_episodes):
    board = create_board()
    data = [] # board, action, log_prob, next_board, reward

    while True:
        # select action and get log_prob
        action, log_prob = policy_player(board, training=True)

        next_board = drop_piece(board, policy_piece, action)
        if check_win(next_board, policy_piece):
            data.append([board, action, log_prob, next_board, rewards["win"]])
            break
        elif len(get_available_cols(next_board)) == 0:
            data.append([board, action, log_prob, next_board, rewards["draw"]])
            break
            
        opponent_action = opponent_player(next_board, opponent_piece)
        next_board = drop_piece(next_board, opponent_piece, opponent_action)
        if check_win(next_board, opponent_piece):
            data.append([board, action, log_prob, next_board, rewards["lose"]])
            break
        elif len(get_available_cols(next_board)) == 0:
            data.append([board, action, log_prob, next_board, rewards["draw"]])
            break

        data.append([board, action, log_prob, next_board, rewards["turn"]])
        board = next_board

    loss = update_policy(optimizer, data)
    loss = loss.detach()
    losses.append(loss)

    if episode % 100 == 0:
        win_rate, avg_moves_to_win = eval_winrate(policy_player, opponent_player)
        eval_history.append([episode+1, win_rate, avg_moves_to_win])
        print(f"Episode {episode}, Loss: {loss:.2f}, Win Rate: {win_rate:.2f}, Avg Moves to Win: {avg_moves_to_win:.2f}")

# save model weights and eval history
torch.save(policy_net, 'policy_cnn.pth')
pd.DataFrame(eval_history, columns=['episode', 'win_rate', 'avg_moves_to_win']).to_csv("./eval_history.csv", index=False)