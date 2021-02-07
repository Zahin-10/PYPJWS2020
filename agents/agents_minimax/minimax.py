from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2, PlayerAction, apply_player_action, connected_four, \
    get_diags_of_matrix
import numpy as np
from typing import Optional

desiredBoardNp = np.full((6, 7), NO_PLAYER)
desiredBoardNp[2, 2] = PLAYER1
desiredBoardNp[2, 3] = PLAYER1
desiredBoardNp[3, 2] = PLAYER2
desiredBoardNp[3, 3] = PLAYER2
desiredBoardNp[3, 4] = PLAYER2
desiredBoardNp[3, 5] = PLAYER2
desiredBoardNp[4, 1] = PLAYER2
desiredBoardNp[4, 2] = NO_PLAYER
desiredBoardNp[4, 3] = NO_PLAYER
desiredBoardNp[4, 4] = NO_PLAYER
desiredBoardNp[5, 1] = PLAYER2
desiredBoardNp[5, 2] = PLAYER2
desiredBoardNp[5, 3] = PLAYER1
desiredBoardNp[5, 4] = PLAYER1


def minimax_decision(board: np.ndarray, player: BoardPiece, maximizing: bool, depth: int) -> PlayerAction:
    possible_actions = np.arange(PlayerAction(0), PlayerAction(7), PlayerAction(1), PlayerAction)
    for op in possible_actions:
        board = apply_player_action(board, op, player, True)
        minimax_value(maximizing, board, depth)
    return PlayerAction(0)


def minimax_value(maximizing: bool, board: np.ndarray, depth: int) -> int:
    if depth == 0:
        return heuristic(maximizing, board)
    elif maximizing:
        value = np.NINF
        return np.amax([value, minimax_decision(board, PLAYER2, True, depth - 1)])
    else:
        value = np.inf
        return np.amax([value, minimax_decision(board, PLAYER1, False, depth - 1)])


def heuristic(maximizing: bool, board: np.ndarray):
    score = 0
    score = score + feature1(board, maximizing)
    return score


def feature1(board: np.ndarray, maximizing: bool):
    if maximizing:
        result = np.inf if connected_four(board, PLAYER2) else 0.0
    else:
        result = np.NINF if connected_four(board, PLAYER1) else 0.0

    return result


def extract_connected(board: np.ndarray, player_to_check: BoardPiece, maximizing: bool,
                      diags: Optional[list] = None) -> bool:
    if diags is None:
        diags = get_diags_of_matrix(board)
    for pawns in diags:
        last_player = None
        match_count = 0
        gap_count = 0
        start_index = None
        end_index = None
        for index, player in enumerate(pawns):
            if last_player != player and player == player_to_check and gap_count != 1:
                start_index = index
                match_count = 1
                gap_count = 0
            elif last_player == player and player == player_to_check:
                match_count += 1
                end_index = index
            elif player == NO_PLAYER:
                gap_count += 1
            last_player = player

    return True


def feature_extract(data: list, gap_count, board: np.ndarray, start_end: tuple, maximizing: bool, row,
                    data_type="horizontal"):
    player_to_check = PLAYER2 if maximizing else PLAYER1
    opponent = PLAYER1 if maximizing else PLAYER2
    if data_type == "diagonal" and row < 5:
        # This is done for checking whether a column is playable in case of diagonals. Since we have to
        # look two row down the row count is increased
        row += 2
    else:
        row += 1 #Since its not a diagonal we only look one row below

    if gap_count == 0 and len(data) > 4:  # Has no gaps in between patterns
        if start_end[0] != 0 and data[start_end[0] - 1] == NO_PLAYER:  # Pattern starts with blanks spaces in the beginning
            if start_end[1] == (len(data) - 1) or data[start_end[1] + 1] == opponent:  # No Blank spaces in the end
                if board[row, start_end[0] - 1] != NO_PLAYER:  # Checks if blanks space in that col is playable or not
                    return 900000 if maximizing else -900000
                else:
                    return 0
            elif data[start_end[1] + 1] == NO_PLAYER:  # Has Blank spaces in the end
                if board[row, start_end[0] - 1] != NO_PLAYER and board[row, start_end[1] + 1] != NO_PLAYER:  # Both blanks spaces are playable
                    return np.inf if maximizing else np.NINF
                elif (board[row, start_end[0] - 1] != NO_PLAYER and board[row, start_end[1] + 1] == NO_PLAYER) or (
                        board[row, start_end[0] - 1] == NO_PLAYER and board[row, start_end[1] + 1] != NO_PLAYER):  # Only one of the blank space is playable
                    return 900000 if maximizing else -900000
                else:
                    return 0
        elif start_end[0] == 0 and data[start_end[1] + 1] == NO_PLAYER:  # Pattern starts from the begining Has Blank spaces in the end
            if board[row, start_end[1] + 1] != NO_PLAYER:
                return 900000 if maximizing else -900000
            else:
                return 0
    elif gap_count == 0 and len(data) == 4:  # Only has 4 spaces for pawns happens to diagonals only
        if (start_end[0] != 0 and data[start_end[0] - 1] == NO_PLAYER) or (
                start_end[0] == 0 and data[start_end[1] + 1] == NO_PLAYER):
            if board[row, start_end[0] - 1] != NO_PLAYER:  # Checks if blanks space in that col is playable or not
                return 900000 if maximizing else -900000
            else:
                return 0
    elif gap_count == 1: #Pattern has gaps in between
        gap_index = None
        if data[start_end[0]+1] == NO_PLAYER:
            gap_index = start_end[0] + 1
        else:
            gap_index = start_end[1] - 1
        if board[row, gap_index] != NO_PLAYER:  # Checks if blanks space in that col is playable or not
            return 900000 if maximizing else -900000
        else:
            return 0
