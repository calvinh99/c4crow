from .base_player import Player
import numpy as np
import c4crow.c4_engine as c4
import time
import sys

class MinimaxPlayer(Player):
    MINIMAX_WIN_SCORE = 1000000

    def __init__(self, max_depth: int = 4, gamma: float = 0.999, xray: bool = False):
        self.max_depth = max_depth
        self.gamma = gamma
        self.xray = xray

    def make_move(self, board: np.ndarray, piece: int) -> int:
        t0 = time.time()
        best_col, best_score = self.minimax_tree_search(board, piece, self.max_depth, True)
        t1 = time.time(); dt = t1 - t0

        if self.xray:
            print(f"\nMinimax search completed in {dt:.2f} seconds")
            print(f"Best column: {best_col}, score: {best_score:.2f}")
            self.print_move_scores(board, piece)

        return best_col

    def minimax_tree_search(self, board: np.ndarray, piece: int, depth: int, maximizing: bool) -> (int, float):
        available_cols = c4.get_available_cols(board)
        if depth == 0 or c4.check_win(board, c4.get_opponent_piece(piece)) != "not done":
            return None, self.evaluate_board(board, piece)
        
        best_col = None
        if maximizing:
            best_score = float('-inf')
        else:
            best_score = float('inf')

        for col in available_cols:
            new_board = c4.drop_piece(board, piece, col)
            _, score = self.minimax_tree_search(new_board, c4.get_opponent_piece(piece), depth - 1, not maximizing)
            score = self.gamma * score
            if maximizing and score > best_score:
                best_score, best_col = score, col
            elif not maximizing and score < best_score:
                best_score, best_col = score, col

        return best_col, best_score

    def evaluate_board(self, board: np.ndarray, piece: int) -> float:
        # helper function for minimax player, but possibly could be used in other players
        opponent_piece = c4.get_opponent_piece(piece)
        if c4.check_win(board, piece) == "win":
            return self.MINIMAX_WIN_SCORE
        elif c4.check_win(board, opponent_piece) == "win":
            return -self.MINIMAX_WIN_SCORE

        score = 0
        # reward center cols
        center_col_idx = c4.N_COLS // 2
        score += 3 * sum(board[:, center_col_idx] == piece)
        # reward consecutive pieces
        n_connections = c4.count_piece_connections(board, piece)
        n_opponent_connections = c4.count_piece_connections(board, opponent_piece)
        for n, count in n_connections.items():
            score += n * count
        for n, count in n_opponent_connections.items():
            score -= n * count # punish for piece connections of opponent
        return score
    
    def print_move_scores(self, board: np.ndarray, piece: int):
        available_cols = c4.get_available_cols(board)
        move_scores = []

        for col in available_cols:
            new_board = c4.drop_piece(board, piece, col)
            _, score = self.minimax_tree_search(new_board, c4.get_opponent_piece(piece), self.max_depth - 1, False)
            move_scores.append((col, score))

        print("\nMove scores:")
        for col, score in move_scores:
            print(f"col {col}: {score:.2f}")
        print()

        self.visualize_board_scores(board, piece, move_scores)

    def visualize_board_scores(self, board: np.ndarray, piece: int, move_scores):
        score_board = np.full(board.shape, '  ')
        for col, score in move_scores:
            row = np.where(board[:, col] == 0)[0][-1]
            if abs(score) > 100000:
                score_str = 'W' if score > 0 else 'L'
            else:
                score_str = f"{score:.2f}"
            score_board[row, col] = f"{score_str:>2}"

        print("\nBoard with move scores (truncated for better visuals):")
        for row in score_board:
            print("|" + "|".join(row) + "|")
        print("-" * (3 * c4.N_COLS + 1))
        print("|" + "|".join([f"{i:2d}" for i in range(c4.N_COLS)]) + "|")
        print()

