from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2, PlayerAction, apply_player_action, connected_four, \
    get_diags_of_matrix
import numpy as np
from typing import Optional

desiredBoardNp = np.full((6, 7), NO_PLAYER)
desiredBoardNp[2, 2] = PLAYER1
desiredBoardNp[2, 3] = PLAYER1
desiredBoardNp[3, 2] = PLAYER2
desiredBoardNp[3, 3] = PLAYER1
desiredBoardNp[3, 4] = PLAYER1
desiredBoardNp[4, 1] = PLAYER2
desiredBoardNp[4, 2] = PLAYER1
desiredBoardNp[4, 3] = PLAYER2
desiredBoardNp[4, 4] = PLAYER2
desiredBoardNp[5, 1] = PLAYER2
desiredBoardNp[5, 2] = PLAYER2
desiredBoardNp[5, 3] = PLAYER1
desiredBoardNp[5, 4] = PLAYER1


def minimax_decision(board: np.ndarray, player: BoardPiece, maximizing: bool, depth: int) -> PlayerAction:
    possible_actions = np.arange(PlayerAction(0), PlayerAction(7), PlayerAction(1), PlayerAction)
    score = np.float(0.0)
    best_action = PlayerAction(4)
    for op in possible_actions:
        board = apply_player_action(board, op, player, True)
        new_score = minimax_value(maximizing, board, depth)
        if (maximizing == True and new_score > score) or (maximizing == False and new_score < score):
            score = new_score
            best_action = op

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
    player_to_check = PLAYER2 if maximizing else PLAYER1
    opponent = PLAYER1 if maximizing else PLAYER2
    score = 0
    score = score + feature_one(board, maximizing)
    score = score + feature_one(board, not maximizing)
    if score == 0.0:
        score += extract_score(board, player_to_check, maximizing)
        score += extract_score(board, opponent, not maximizing)
    return score


def feature_one(board: np.ndarray, maximizing: bool):
    if maximizing:
        result = np.inf if connected_four(board, PLAYER2) else 0.0
    else:
        result = np.NINF if connected_four(board, PLAYER1) else 0.0

    return result


def extract_score(board: np.ndarray, player_to_check: BoardPiece, maximizing: bool,
                  diags: Optional[list] = None) -> float:
    score = 0
    if diags is None:
        diags = get_diags_of_matrix(board)
    score += extract_connected(diags[:6], player_to_check, board, maximizing)
    score += extract_connected(diags[6:], player_to_check, board, maximizing)
    for data in board:
        score += extract_connected(data, player_to_check, board, maximizing, "horizontal")
    for data in board.T:
        score += extract_connected(data, player_to_check, board, maximizing, "vertical")

    return score


def extract_connected(data: list, player_to_check: BoardPiece, board, maximizing, row_type="diagonal"):
    score = 0
    for row, pawns in enumerate(data):
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
        if row < 4:
            row += 3
        else:
            row = 5
        if match_count == 3:
            score += feature_two(pawns, gap_count, board, (start_index, end_index), maximizing, row, row_type)
        elif match_count == 2:
            score += feature_three(pawns, board, (start_index, end_index), maximizing, row, row_type)
        elif match_count == 1:
            score += feature_four(pawns, maximizing)
    return score


def feature_two(data: list, gap_count, board: np.ndarray, start_end: tuple, maximizing: bool, row,
                data_type="horizontal"):
    player_to_check = PLAYER2 if maximizing else PLAYER1
    opponent = PLAYER1 if maximizing else PLAYER2
    if data_type == "vertical":
        if data[start_end[1] + 1] == NO_PLAYER:
            return 900000 if maximizing else -900000
        else:
            return 0
    if data_type == "diagonal" and row < 5:
        # This is done for checking whether a column is playable in case of diagonals. Since we have to
        # look two row down the row count is increased
        row += 2
    else:
        row += 1  # Since its not a diagonal we only look one row below

    if gap_count == 0 and len(data) > 4:  # Has no gaps in between patterns
        if start_end[0] != 0 and data[
            start_end[0] - 1] == NO_PLAYER:  # Pattern starts with blanks spaces in the beginning
            if start_end[1] == (len(data) - 1) or data[start_end[1] + 1] == opponent:  # No Blank spaces in the end
                if board[row, start_end[0] - 1] != NO_PLAYER:  # Checks if blanks space in that col is playable or not
                    return 900000 if maximizing else -900000
                else:
                    return 0
            elif data[start_end[1] + 1] == NO_PLAYER:  # Has Blank spaces in the end
                if board[row, start_end[0] - 1] != NO_PLAYER and board[
                    row, start_end[1] + 1] != NO_PLAYER:  # Both blanks spaces are playable
                    return np.inf if maximizing else np.NINF
                elif (board[row, start_end[0] - 1] != NO_PLAYER and board[row, start_end[1] + 1] == NO_PLAYER) or (
                        board[row, start_end[0] - 1] == NO_PLAYER and board[
                    row, start_end[1] + 1] != NO_PLAYER):  # Only one of the blank space is playable
                    return 900000 if maximizing else -900000
                else:
                    return 0
        elif start_end[0] == 0 and data[
            start_end[1] + 1] == NO_PLAYER:  # Pattern starts from the begining Has Blank spaces in the end
            if board[row, start_end[1] + 1] != NO_PLAYER:
                return 900000 if maximizing else -900000
            else:
                return 0
    elif gap_count == 0 and len(data) == 4:  # Only has 4 spaces for pawns happens to diagonals only
        if start_end[0] != 0 and data[start_end[0] - 1] == NO_PLAYER:
            if board[row, start_end[0] - 1] != NO_PLAYER:  # Checks if blanks space in that col is playable or not
                return 900000 if maximizing else -900000
            else:
                return 0
        elif start_end[0] == 0 and data[start_end[1] + 1] == NO_PLAYER:
            if board[row, start_end[1] + 1] != NO_PLAYER:  # Checks if blanks space in that col is playable or not
                return 900000 if maximizing else -900000
            else:
                return 0
    elif gap_count == 1:  # Pattern has gaps in between
        gap_index = None
        if data[start_end[0] + 1] == NO_PLAYER:
            gap_index = start_end[0] + 1
        else:
            gap_index = start_end[1] - 1
        if board[row, gap_index] != NO_PLAYER:  # Checks if blanks space in that col is playable or not
            return 900000 if maximizing else -900000
        else:
            return 0
    return 0


def feature_three(data: list, board: np.ndarray, start_end: tuple, maximizing: bool, row, data_type="horizontal"):
    available_spots = 0
    player_to_check = PLAYER2 if maximizing else PLAYER1
    opponent = PLAYER1 if maximizing else PLAYER2
    if data_type == "vertical":
        if data[start_end[1] + 1] == NO_PLAYER:
            return 10000 if maximizing else -10000
        else:
            return 0
    if data_type == "diagonal" and row < 5:
        # This is done for checking whether a column is playable in case of diagonals. Since we have to
        # look two row down thats why the row count is increased
        row += 2
    else:
        row += 1  # Since its not a diagonal we only look one row below

    if len(data) > 4:
        if start_end[0] != 0 and data[
            start_end[0] - 1] == NO_PLAYER:  # Pattern starts with blanks spaces in the beginning
            if start_end[1] == (len(data) - 1) or data[start_end[1] + 1] == opponent:  # No Blank spaces in the end
                for i in range(start_end[0] - 1, -1, -1):
                    if board[row, i] != NO_PLAYER:  # Checks if blanks space in that col is playable or not
                        available_spots += 1
                    else:
                        break
            elif data[start_end[1] + 1] == NO_PLAYER:  # Has Blank spaces in the end
                if board[row, start_end[0] - 1] != NO_PLAYER and board[
                    row, start_end[1] + 1] != NO_PLAYER:  # Both blanks spaces are playable
                    for i in range(start_end[0] - 1, -1, -1):
                        if board[row, i] != NO_PLAYER:  # Checks if blanks space in that col is playable or not
                            available_spots += 1
                        else:
                            break
                    for i in range(start_end[1] + 1, len(data)):
                        if board[row, i] != NO_PLAYER:  # Checks if blanks space in that col is playable or not
                            available_spots += 1
                        else:
                            break
        elif start_end[0] == 0 and data[
            start_end[1] + 1] == NO_PLAYER:  # Pattern starts from the begining Has Blank spaces in the end
            for i in range(start_end[1] + 1, len(data)):
                if board[row, i] != NO_PLAYER:  # Checks if blanks space in that col is playable or not
                    available_spots += 1
                else:
                    break
    elif len(data) == 4:
        if start_end[0] != 0 and data[start_end[0] - 1] == NO_PLAYER:
            if board[row, start_end[0] - 1] != NO_PLAYER:  # Checks if blanks space in that col is playable or not
                available_spots += 1
        elif start_end[0] == 0 and data[start_end[1] + 1] == NO_PLAYER:
            if board[row, start_end[1] + 1] != NO_PLAYER:  # Checks if blanks space in that col is playable or not
                available_spots += 1

    if available_spots == 2:
        return 10000 if maximizing else -10000
    elif available_spots == 3:
        return 20000 if maximizing else -20000
    elif available_spots == 4:
        return 30000 if maximizing else -30000
    elif available_spots == 5:
        return 40000 if maximizing else -40000
    else:
        return 0


def feature_four(data: list, maximizing: bool):
    player_to_check = PLAYER2 if maximizing else PLAYER1
    player_indices = [i for i, x in enumerate(data) if x == player_to_check]
    score = 0
    for pos in player_indices:
        if pos == 3:
            score += 200
        elif pos == 0 or pos == 6:
            score += 40
        elif pos == 1 or pos == 5:
            score += 70
        elif pos == 2 or pos == 4:
            score += 120
    if not maximizing:
        score = score * (-1)
    return score

minimax_decision(desiredBoardNp, PLAYER1, False)