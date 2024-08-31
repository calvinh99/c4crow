import numpy as np
from enum import Enum
from typing import Optional, Dict, Tuple
from collections import deque, defaultdict

N_ROWS = 6
N_COLS = 7
TOP_ROW = 0
BOTTOM_ROW = N_ROWS - 1

YELLOW = "\033[1;33m"
RED = "\033[1;31m"
RESET = "\033[0m"
BOLD = "\033[1m"
EMPTY = 0
P1 = 1
P2 = 2

def get_opponent_piece(piece: int):
    if piece == 1: return 2
    elif piece == 2: return 1

def create_board():
    return np.zeros([N_ROWS, N_COLS], dtype=np.int8)

def make_board_hashable(board: np.ndarray) -> str:
    return ''.join(map(str, board.flatten()))

def string_to_board(board_string: str) -> np.ndarray:
    return np.array(list(map(int, board_string))).reshape(6, 7)

def get_available_cols(board):
    return np.where(board[TOP_ROW] == EMPTY)[0]

def double_channel_one_hot_board(board: np.ndarray, player_piece: int) -> np.ndarray:
    # Create two 6x7 boards: first for player pieces (1s), second for opponent pieces (1s). All other positions are 0s.
    opponent_piece = get_opponent_piece(player_piece)
    channels = np.zeros((2, N_ROWS, N_COLS), dtype=np.float32)
    channels[0] = (board == player_piece)
    channels[1] = (board == opponent_piece)
    return channels

def drop_piece(board: np.ndarray, piece: int, col_idx: int) -> Optional[np.ndarray]:
    if col_idx not in get_available_cols(board):
        print(f"Column {col_idx} is not available!")
        return None
    board = board.copy()
    row_idx = np.sum([board[:, col_idx] == EMPTY]) - 1 # so if 4 empty in the 6 rows, then we place at 3
    board[row_idx, col_idx] = piece
    return board
    
def check_win(board: np.ndarray, piece: int) -> str:
    if len(get_available_cols(board)) == 0:
        return "draw"

    for row_idx in range(N_ROWS):
        for col_idx in range(N_COLS):
            axis_results = check_axis(board, row_idx, col_idx, piece)
            
            # If any axis has 4 or more connected pieces, it's a win
            if any(n >= 4 for n in axis_results.keys()):
                return "win"
    
    return "not done"

def count_piece_connections(board: np.ndarray, piece: int) -> Dict[int, int]:
    connections = defaultdict(set)
    
    for row_idx in range(N_ROWS):
        for col_idx in range(N_COLS):
            if board[row_idx, col_idx] == piece:
                results = check_axis(board, row_idx, col_idx, piece)
                for n, cell_strings in results.items():
                    connections[n].update(cell_strings)
    
    return {n: len(cell_strings) for n, cell_strings in connections.items()}

def check_axis(board: np.ndarray, row_idx: int, col_idx: int, piece: int) -> dict:
    if board[row_idx][col_idx] != piece:
        return {} # none are connected
    
    result = defaultdict(list)

    # check all axis vectors
    # all cell strings should go from L -> R, and U -> D, with L -> R taking priority
    for rd, cd in [
        (0, 1), # horizontal axis
        (1, 0), # vertical axis
        (1, 1), # positive diagonal axis
        (-1, 1), # negative diagonal axis
    ]:
        # our starting piece is valid
        n_connected = 1
        cell_string = deque([(row_idx, col_idx)])
    
        # check forward dir
        r, c = row_idx + rd, col_idx + cd
        while (0 <= r <= N_ROWS-1) and (0 <= c <= N_COLS-1) and (board[r, c] == piece):
            n_connected += 1
            cell_string.append((r, c))
            r, c = r + rd, c + cd
        
        # check reverse dir
        rd, cd = -rd, -cd
        r, c = row_idx + rd, col_idx + cd
        while (0 <= r <= N_ROWS-1) and (0 <= c <= N_COLS-1) and (board[r, c] == piece):
            n_connected += 1
            cell_string.appendleft((r, c)) # prepend, since it's before the starting piece and we want L -> R or U -> D
            r, c = r + rd, c + cd
        
        result[n_connected].append(str(list(cell_string))) # e.g. { 4: "[(6, 1), (6, 2), (6, 3), (6, 4)]", ... }
    
    return result

def display_board(board):
    rows, cols = board.shape
    print(BOLD + "    " + "   ".join(str(i) for i in range(cols)) + RESET)
    print(BOLD + "  ╔" + "═══╤" * (cols - 1) + "═══╗" + RESET)
    for r in range(rows):
        row_str = BOLD + f"{r} ║" + RESET
        for c in range(cols):
            if board[r, c] == EMPTY:
                row_str += f" · "
            elif board[r, c] == P1:
                row_str += f" {YELLOW}O{RESET} "
            elif board[r, c] == P2:
                row_str += f" {RED}X{RESET} "
            if c < cols - 1:
                row_str += BOLD + "│" + RESET
        row_str += BOLD + "║" + RESET
        print(row_str)
        if r < rows - 1:
            print(BOLD + "  ╟" + "───┼" * (cols - 1) + "───╢" + RESET)
    print(BOLD + "  ╚" + "═══╧" * (cols - 1) + "═══╝" + RESET)