import pytest
import numpy as np
from c4crow.c4_engine import Pieces, create_board, get_available_cols, drop_piece, check_win, check_axis

def board_to_string(board):
    return '\n'.join([' '.join(map(str, row)) for row in board])

def test_get_available_cols():
    # Test with empty board
    board = create_board()
    assert list(get_available_cols(board)) == list(range(7)), \
        f"Empty board should have all columns available.\nBoard state:\n{board_to_string(board)}"

    # Test with full board
    full_board = np.full((6, 7), Pieces.P1.value)
    assert list(get_available_cols(full_board)) == [], \
        f"Full board should have no columns available.\nBoard state:\n{board_to_string(full_board)}"

    # Test with partially filled board
    partial_board = create_board()
    partial_board[:, 0] = Pieces.P1.value  # Fill first column
    partial_board[5:3:-1, 1] = Pieces.P2.value  # Fill bottom 3 of second column
    expected = [np.int64(i) for i in range(1, 7)]  # Columns 1-6 as np.int64
    assert list(get_available_cols(partial_board)) == expected, \
        f"Partially filled board should have columns 1-6 available.\nBoard state:\n{board_to_string(partial_board)}"

def test_drop_piece():
    board = create_board()

    # Test dropping in empty column
    new_board = drop_piece(board, Pieces.P1, 0)
    assert new_board is not None, "Dropping piece in empty column should not return None"
    assert new_board[5, 0] == Pieces.P1.value, \
        f"Piece not correctly placed at bottom of empty column.\nBoard state:\n{board_to_string(new_board)}"
    assert new_board is not board, "drop_piece should return a new board, not modify the existing one"

    # Test dropping in partially filled column
    new_board = drop_piece(new_board, Pieces.P2, 0)
    assert new_board is not None, "Dropping piece in partially filled column should not return None"
    assert new_board[4, 0] == Pieces.P2.value, \
        f"Piece not correctly placed in partially filled column.\nBoard state:\n{board_to_string(new_board)}"

    # Test dropping in full column
    full_col_board = create_board()
    full_col_board[:, 0] = Pieces.P1.value
    assert drop_piece(full_col_board, Pieces.P2, 0) is None, \
        f"Dropping piece in full column should return None.\nBoard state:\n{board_to_string(full_col_board)}"

    # Test dropping in invalid column
    assert drop_piece(board, Pieces.P1, 7) is None, "Dropping piece in invalid column (7) should return None"
    assert drop_piece(board, Pieces.P1, -1) is None, "Dropping piece in invalid column (-1) should return None"

def test_check_win():
    board = create_board()

    # Test horizontal win
    board[5, :4] = Pieces.P1.value
    assert check_win(board, Pieces.P1), \
        f"Horizontal win not detected.\nBoard state:\n{board_to_string(board)}"

    # Test vertical win
    board = create_board()
    board[:4, 0] = Pieces.P2.value
    assert check_win(board, Pieces.P2), \
        f"Vertical win not detected.\nBoard state:\n{board_to_string(board)}"

    # Test diagonal win (positive slope)
    board = create_board()
    for i in range(4):
        board[5-i, i] = Pieces.P1.value
    assert check_win(board, Pieces.P1), \
        f"Diagonal win (positive slope) not detected.\nBoard state:\n{board_to_string(board)}"

    # Test diagonal win (negative slope)
    board = create_board()
    for i in range(4):
        board[i, i] = Pieces.P2.value
    assert check_win(board, Pieces.P2), \
        f"Diagonal win (negative slope) not detected.\nBoard state:\n{board_to_string(board)}"

    # Test near-win scenario
    board = create_board()
    board[5, :3] = Pieces.P1.value
    assert not check_win(board, Pieces.P1), \
        f"Near-win scenario incorrectly detected as win.\nBoard state:\n{board_to_string(board)}"

    # Test empty board
    assert not check_win(create_board(), Pieces.P1), \
        "Empty board incorrectly detected as win"

    # Test full board with no win
    full_board = np.array([
        [1, 2, 1, 2, 1, 2, 1],
        [1, 2, 1, 2, 1, 2, 1],
        [2, 1, 2, 1, 2, 1, 2],
        [2, 1, 2, 1, 2, 1, 2],
        [1, 2, 1, 2, 1, 2, 1],
        [1, 2, 1, 2, 1, 2, 1]
    ])
    assert not check_win(full_board, Pieces.P1), \
        f"Full board with no win incorrectly detected as win for P1.\nBoard state:\n{board_to_string(full_board)}"
    assert not check_win(full_board, Pieces.P2), \
        f"Full board with no win incorrectly detected as win for P2.\nBoard state:\n{board_to_string(full_board)}"

def test_check_axis():
    board = create_board()

    # Test horizontal connections
    board[5, :5] = Pieces.P1.value
    result = check_axis(board, 5, 0, Pieces.P1)
    assert 5 in result, f"Horizontal connection of 5 not detected.\nBoard state:\n{board_to_string(board)}\nResult: {result}"
    assert 4 not in result, f"Unexpected horizontal connection of 4 detected.\nBoard state:\n{board_to_string(board)}\nResult: {result}"

    # Test vertical connections
    board = create_board()
    board[:4, 0] = Pieces.P2.value
    result = check_axis(board, 3, 0, Pieces.P2)
    assert 4 in result, f"Vertical connection of 4 not detected.\nBoard state:\n{board_to_string(board)}\nResult: {result}"

    # Test diagonal connections (positive slope)
    board = create_board()
    for i in range(4):
        board[5-i, i] = Pieces.P1.value
    result = check_axis(board, 5, 0, Pieces.P1)
    assert 4 in result, f"Diagonal connection (positive slope) of 4 not detected.\nBoard state:\n{board_to_string(board)}\nResult: {result}"

    # Test diagonal connections (negative slope)
    board = create_board()
    for i in range(4):
        board[i, i] = Pieces.P2.value
    result = check_axis(board, 0, 0, Pieces.P2)
    assert 4 in result, f"Diagonal connection (negative slope) of 4 not detected.\nBoard state:\n{board_to_string(board)}\nResult: {result}"

    # Test no connections
    board = create_board()
    result = check_axis(board, 0, 0, Pieces.P1)
    assert result == {}, f"Empty board incorrectly detected connections.\nBoard state:\n{board_to_string(board)}\nResult: {result}"

    # Test at board edges
    board = create_board()
    board[0, :4] = Pieces.P1.value
    result = check_axis(board, 0, 3, Pieces.P1)
    assert 4 in result, f"Connection at board edge not detected.\nBoard state:\n{board_to_string(board)}\nResult: {result}"