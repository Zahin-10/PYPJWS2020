from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2, PlayerAction, apply_player_action, connected_four, \
    get_diags_of_matrix, valid_move, SavedState
import numpy as np
from typing import Optional

maximizing_player = NO_PLAYER
minimizing_player = NO_PLAYER


def get_agent_move(board: np.ndarray, _player: BoardPiece, saved_state: Optional[SavedState], depth=4):
    global maximizing_player
    global minimizing_player
    minimizing_player = _player
    maximizing_player = PLAYER2 if _player == PLAYER1 else PLAYER1
    action, score = minimax_decision(board, _player, False, depth)
    return PlayerAction(action), saved_state


def minimax_decision(board_: np.ndarray, player: BoardPiece, maximizing: bool, depth: int) -> (PlayerAction, int):
    possible_actions = valid_move(board_)
    score = np.float("0")
    best_action = PlayerAction(4)
    board = board_.copy()
    if depth == 0:
        return None, heuristic(maximizing, board)
    elif maximizing:
        value = np.NINF
        for op in possible_actions:
            try:
                new_board = apply_player_action(board, PlayerAction(op), player, True)
                new_score = minimax_decision(new_board, minimizing_player, False, depth - 1)[1]
                if new_score > value:
                    value = new_score
                    best_action = op
            except Exception as error:
                continue
        return best_action, value
    else:
        value = np.inf
        for op in possible_actions:
            try:
                new_board = apply_player_action(board, PlayerAction(op), player, True)
                new_score = minimax_decision(new_board, maximizing_player, True, depth - 1)[1]
                if new_score < value:
                    value = new_score
                    best_action = op
            except Exception as error:
                continue
        return best_action, value


def heuristic(maximizing: bool, board: np.ndarray):
    """
    Heuristic function firstly looks for different features(4 possiblefeatures in this algorithm, namely feature_one,
    feature_two, feature_three and feature_four) on the board and then gives them proper values. Finally, the
    heuristic function returns a summation of all the values of features on the chess
    board
    """
    player_to_check = maximizing_player if maximizing else minimizing_player
    opponent = minimizing_player if maximizing else maximizing_player

    score = 0
    score = score + feature_one(board, maximizing)
    score = score + feature_one(board, not maximizing)
    if score == 0.0:
        score += extract_score(board, player_to_check, maximizing)
        score += extract_score(board, opponent, not maximizing)
    return score


def feature_one(board: np.ndarray, maximizing: bool):
    
    """
    Checks whether A move can be made on either
    immediately adjacent columns and see if Four chessmen are connected horizontally, vertically or diagonally
    If yes, this specific game state is given infinite point for maximizing agent and negative infinite to the
    minimizing agent.
    """
    
    if maximizing:
        result = np.inf if connected_four(board, maximizing_player) else 0.0
    else:
        result = np.NINF if connected_four(board, minimizing_player) else 0.0

    return result


def extract_score(board: np.ndarray, player_to_check: BoardPiece, maximizing: bool,
                  diags: Optional[list] = None) -> float:
    
    """

    This function takes into account all the possible features from the 4 features
    in a given state of the game and returns the total score.

    """
    
    score = 0
    flipped_board = np.fliplr(board)
    if diags is None:
        diags = get_diags_of_matrix(board)
        diags_flipped = get_diags_of_matrix(flipped_board)
    score += extract_connected(diags[:6], player_to_check, board, maximizing)
    score += extract_connected(diags_flipped[:6], player_to_check, flipped_board, maximizing)

    score += extract_connected(board, player_to_check, board, maximizing, "horizontal")
    transposed_flipped_board = np.flip(board.T) # This is done to extract score for vertically connected pawns
    score += extract_connected(transposed_flipped_board, player_to_check, board, maximizing, "vertical")

    return score


def extract_connected(data, player_to_check: BoardPiece, board, maximizing, row_type="diagonal"):
    score = 0
    for row, pawns in enumerate(data):
        last_player = None
        match_count = 0
        gap_count = 0
        start_index = None
        end_index = None
        for index, player in enumerate(pawns):
            if last_player != player and player == player_to_check and (match_count == 0 or gap_count != 1):
                start_index = index
                match_count = 1
                gap_count = 0
            elif last_player == player and player == player_to_check:
                match_count += 1
                end_index = index
            elif player == NO_PLAYER and (last_player == NO_PLAYER or last_player == player_to_check):
                gap_count += 1
            else:
                gap_count = 0
            last_player = player
        if row_type == "diagonal":
            if row < 3:
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
    """
    Checks whether three chessmen are connected horizontally, vertically or diagonally and give points +900000 for
    maximizing state and -900000 for minimizing state.
    """
    player_to_check = maximizing_player if maximizing else minimizing_player
    opponent = minimizing_player if maximizing else maximizing_player
    if data_type == "vertical":
        if start_end[1] != len(data) - 1:
            if data[start_end[1] + 1] == NO_PLAYER:
                return 900000 if maximizing else -900000
        return 0

    if gap_count == 0 and len(data) > 4:  # Has no gaps in between patterns
        if start_end[0] != 0 and data[start_end[0] - 1] == NO_PLAYER:  # Pattern starts with blanks spaces in the beginning
            if start_end[1] == (len(data) - 1) or data[start_end[1] + 1] == opponent:  # No Blank spaces in the end
                if data_type == "diagonal":
                    blank_pos = row - start_end[0] + 1
                else:
                    blank_pos = row
                if can_play_spot((blank_pos, start_end[0] - 1), board):  # Checks if blanks space in that col is playable or not
                    return 900000 if maximizing else -900000
                else:
                    return 0
            elif data[start_end[1] + 1] == NO_PLAYER:  # Has Blank spaces in the end
                if data_type == "diagonal":
                    blank_pos = row - start_end[0] + 1
                    blank_pos_end = row - start_end[1] - 1
                else:
                    blank_pos = row
                    blank_pos_end = row
                if can_play_spot((blank_pos, start_end[0] - 1), board) and can_play_spot((blank_pos_end, start_end[1] + 1), board):  # Both blanks spaces are playable
                    return np.inf if maximizing else np.NINF
                elif (can_play_spot((blank_pos, start_end[0] - 1), board) and not can_play_spot((blank_pos_end, start_end[1] + 1), board)) \
                    or (not can_play_spot((blank_pos, start_end[0] - 1), board) and can_play_spot((blank_pos_end, start_end[1] + 1), board)):  # Only one of the blank space is playable
                    return 900000 if maximizing else -900000
                else:
                    return 0
        elif start_end[0] == 0 and data[start_end[1] + 1] == NO_PLAYER:  # Pattern starts from the begining Has Blank spaces in the end
            if data_type == "diagonal":
                blank_pos_end = row - start_end[1] - 1
            else:
                blank_pos_end = row
            if can_play_spot((blank_pos_end, start_end[1] + 1), board):
                return 900000 if maximizing else -900000
            else:
                return 0
    elif gap_count == 0 and len(data) == 4:  # Only has 4 spaces for pawns happens to diagonals only
        if start_end[0] != 0 and data[start_end[0] - 1] == NO_PLAYER:
            blank_pos = row - start_end[0] + 1
            if can_play_spot((blank_pos, start_end[0] - 1), board):  # Checks if blanks space in that col is playable or not
                return 900000 if maximizing else -900000
            else:
                return 0
        elif start_end[0] == 0 and data[start_end[1] + 1] == NO_PLAYER:
            blank_pos_end = row - start_end[1] - 1
            if can_play_spot((blank_pos_end, start_end[1] + 1), board):  # Checks if blanks space in that col is playable or not
                return 900000 if maximizing else -900000
            else:
                return 0
    elif gap_count == 1:  # Pattern has gaps in between
        gap_index = None
        if data[start_end[0] + 1] == NO_PLAYER:
            gap_index = start_end[0] + 1
        else:
            gap_index = start_end[1] - 1
        if data_type == "diagonal":
            blank_pos = row - gap_index + 1
        else:
            blank_pos = row
        if can_play_spot((blank_pos, gap_index), board):  # Checks if blanks space in that col is playable or not
            return 900000 if maximizing else -900000
        else:
            return 0
    return 0


def feature_three(data: list, board: np.ndarray, start_end: tuple, maximizing: bool, row, data_type="horizontal"):
    """
    Checks whether two chessmen are connected horizontally, vertically or diagonally. and if move can only be made on one of
    the immediately adjacent columns. (The value depends on the number of available squares along the direction
    till an unavailable square is met.)

    +ve values are for maximizing state and negative values are for minimizing state

    Number of available squares    Values
              5                   40,000
              4                   30,000
              3                   20,000
              2                   10,000

    """
    available_spots = 0
    player_to_check = maximizing_player if maximizing else minimizing_player
    opponent = minimizing_player if maximizing else maximizing_player
    if data_type == "vertical":
        if start_end[1] != len(data) - 1:
            if data[start_end[1] + 1] == NO_PLAYER:
                return 10000 if maximizing else -10000
        return 0

    if len(data) > 4:
        if start_end[0] != 0 and data[start_end[0] - 1] == NO_PLAYER:  # Pattern starts with blanks spaces in the beginning
            if start_end[1] == (len(data) - 1) or data[start_end[1] + 1] == opponent:  # No Blank spaces in the end
                for i in range(start_end[0] - 1, -1, -1):
                    if data_type == "diagonal":
                        blank_pos = row - i + 1
                    else:
                        blank_pos = row
                    if can_play_spot((blank_pos, i), board):  # Checks if blanks space in that col is playable or not
                        available_spots += 1
                    else:
                        break
            elif data[start_end[1] + 1] == NO_PLAYER:  # Has Blank spaces in the end

                for i in range(start_end[0] - 1, -1, -1):
                    if data_type == "diagonal":
                        blank_pos = row - i + 1
                    else:
                        blank_pos = row
                    if can_play_spot((blank_pos, i), board):  # Checks if blanks space in that col is playable or not
                        available_spots += 1
                    else:
                        break
                for i in range(start_end[1] + 1, len(data)):
                    if data_type == "diagonal":
                        blank_pos_end = row - i - 1
                    else:
                        blank_pos_end = row
                    if can_play_spot((blank_pos_end, i), board):  # Checks if blanks space in that col is playable or not
                        available_spots += 1
                    else:
                        break

        elif start_end[0] == 0 and data[start_end[1] + 1] == NO_PLAYER:  # Pattern starts from the begining Has Blank spaces in the end
            for i in range(start_end[1] + 1, len(data)):
                if data_type == "diagonal":
                    blank_pos_end = row - i - 1
                else:
                    blank_pos_end = row
                if can_play_spot((blank_pos_end, i), board):  # Checks if blanks space in that col is playable or not
                    available_spots += 1
                else:
                    break
    elif len(data) == 4:
        if start_end[0] != 0 and data[start_end[0] - 1] == NO_PLAYER:
            blank_pos = row - start_end[0] + 1
            if can_play_spot((blank_pos, start_end[0] - 1), board):  # Checks if blanks space in that col is playable or not
                available_spots += 1
        elif start_end[0] == 0 and data[start_end[1] + 1] == NO_PLAYER:
            blank_pos_end = row - start_end[1] - 1
            if can_play_spot((blank_pos_end, start_end[1] + 1), board):  # Checks if blanks space in that col is playable or not
                available_spots += 1

    if available_spots == 2:
        return 10000 if maximizing else -10000
    elif available_spots == 3:
        return 20000 if maximizing else -20000
    elif available_spots == 4:
        return 30000 if maximizing else -30000
    elif available_spots >= 5:
        return 40000 if maximizing else -40000
    else:
        return 0


def feature_four(data: list, maximizing: bool):
    """
    Checks if a chessman that is not connected to another same chessman horizontally, vertically
    or diagonally as shown gives the below values according to the given states

    +ve values are for maximizing state and negative values are for minimizing state

    In column d         200
    In column a or g    40
    In column b or f    70
    In column c or e    120

    """
    player_to_check = maximizing_player if maximizing else minimizing_player
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


def can_play_spot(pos:tuple, board: np.ndarray) -> bool:
    pos_x = pos[0]
    if pos[0] == 6:
        pos_x = 5
    elif pos[0] < 0:
        pos_x = 0

    if pos[0] < 5:
        pos_x = pos[0] + 1
        return board[pos_x, pos[1]] != NO_PLAYER
    else:
        return board[pos_x, pos[1]] == NO_PLAYER
