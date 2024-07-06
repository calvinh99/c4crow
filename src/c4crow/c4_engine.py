import numpy as np
from enum import Enum
from typing import Optional, Dict
from collections import deque, defaultdict

N_ROWS = 6
N_COLS = 7
TOP_ROW = 0
BOTTOM_ROW = N_ROWS - 1

YELLOW = "\033[1;33m"
RED = "\033[1;31m"
RESET = "\033[0m"
BOLD = "\033[1m"

class Pieces(Enum):
    EMPTY = (0, "·")
    P1 = (1, f"{YELLOW}O{RESET}")
    P2 = (2, f"{RED}X{RESET}")

    def __init__(self, value, display_str):
        self._value_ = value
        self.display_str = display_str

    def get_opponent_piece(self):
        if self == Pieces.P1:
            return Pieces.P2
        elif self == Pieces.P2:
            return Pieces.P1
        else:
            return Pieces.EMPTY

def create_board():
    return np.zeros([N_ROWS, N_COLS], dtype=np.int8)

def get_available_cols(board):
    return np.where(board[TOP_ROW] == Pieces.EMPTY.value)[0]

# we should offload saving the timesteps (state_0, col_idx, state_1, reward) to the RL training code!
def drop_piece(board: np.ndarray, piece: Pieces, col_idx: int) -> Optional[np.ndarray]:
    if col_idx not in get_available_cols(board):
        print(f"Column {col_idx} is not available!")
        return None
    board = board.copy()
    row_idx = np.sum([board[:, col_idx] == Pieces.EMPTY.value]) - 1 # so if 4 empty in the 6 rows, then we place at 3
    board[row_idx, col_idx] = piece._value_
    return board
    
def check_win(board: np.ndarray, piece: Pieces) -> bool:
    for row_idx in range(N_ROWS):
        for col_idx in range(N_COLS):
            axis_results = check_axis(board, row_idx, col_idx, piece)
            
            # If any axis has 4 or more connected pieces, it's a win
            if any(n >= 4 for n in axis_results.keys()):
                return True
    
    return False

def count_piece_connections(board: np.ndarray, piece: Pieces) -> Dict[int, int]:
    connections = defaultdict(set)
    
    for row_idx in range(N_ROWS):
        for col_idx in range(N_COLS):
            if board[row_idx, col_idx] == piece.value:
                results = check_axis(board, row_idx, col_idx, piece)
                for n, cell_strings in results.items():
                    connections[n].update(cell_strings)
    
    return {n: len(cell_strings) for n, cell_strings in connections.items()}

def check_axis(board: np.ndarray, row_idx: int, col_idx: int, piece: Pieces) -> dict:
    if board[row_idx][col_idx] != piece._value_:
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
        while (0 <= r <= N_ROWS-1) and (0 <= c <= N_COLS-1) and (board[r, c] == piece._value_):
            n_connected += 1
            cell_string.append((r, c))
            r, c = r + rd, c + cd
        
        # check reverse dir
        rd, cd = -rd, -cd
        r, c = row_idx + rd, col_idx + cd
        while (0 <= r <= N_ROWS-1) and (0 <= c <= N_COLS-1) and (board[r, c] == piece._value_):
            n_connected += 1
            cell_string.appendleft((r, c)) # prepend, since it's before the starting piece and we want L -> R or U -> D
            r, c = r + rd, c + cd
        
        result[n_connected].append(str(list(cell_string))) # e.g. { 4: "[(6, 1), (6, 2), (6, 3), (6, 4)]", ... }
    
    return result

def display_board(board):
    rows, cols = board.shape
    
    # Print column numbers
    print(BOLD + "    " + "   ".join(str(i) for i in range(cols)) + RESET)

    # Print top border
    print(BOLD + "  ╔" + "═══╤" * (cols - 1) + "═══╗" + RESET)
    
    for r in range(rows):
        row_str = BOLD + f"{r} ║" + RESET
        for c in range(cols):
            if board[r, c] == Pieces.EMPTY._value_:
                row_str += f" {Pieces.EMPTY.display_str} "
            elif board[r, c] == Pieces.P1._value_:
                row_str += f" {Pieces.P1.display_str} "
            elif board[r, c] == Pieces.P2._value_:
                row_str += f" {Pieces.P2.display_str} "
            
            if c < cols - 1:
                row_str += BOLD + "│" + RESET
        
        row_str += BOLD + "║" + RESET
        print(row_str)
        
        if r < rows - 1:
            print(BOLD + "  ╟" + "───┼" * (cols - 1) + "───╢" + RESET)
    
    # Print bottom border
    print(BOLD + "  ╚" + "═══╧" * (cols - 1) + "═══╝" + RESET)