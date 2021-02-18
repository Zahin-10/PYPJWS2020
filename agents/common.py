from enum import Enum
from typing import Optional
from typing import Callable, Tuple
import numpy as np

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

PlayerAction = np.int8  # The column to be played


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    init = np.full((6, 7), NO_PLAYER)
    return init


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output:
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """
    draw = "|==============|\n"
    for row in board:
        draw = draw + "|"
        for col in row:
            if col == NO_PLAYER:
                draw = draw + "  "
            elif col == PLAYER1:
                draw = draw + "X "
            elif col == PLAYER2:
                draw = draw + "O "
        draw = draw + "|\n"
    draw = draw + "|==============|\n"
    draw = draw + "|0 1 2 3 4 5 6 |"
    print(draw)
    return draw


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    board = initialize_game_state()
    boardStrArr = pp_board.split('\n')
    boardStrArr = boardStrArr[1:len(boardStrArr) - 2]
    for index, row in enumerate(boardStrArr):
        row = row.replace('|', '')
        for colindex in range(0, len(row)):
            if colindex == 0 or (colindex % 2) == 0:
                boardCol = int(colindex / 2)
                if row[colindex] == "X":
                    board[index, boardCol] = PLAYER1
                elif row[colindex] == "O":
                    board[index, boardCol] = PLAYER2

    return board


def apply_player_action(
        board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """
    for index, item in enumerate(board[:, action]):
        if item != NO_PLAYER and index != 0:
            if copy:
                boardCopy = board.copy()
                boardCopy[index - 1, action] = player
                return boardCopy
            else:
                board[index - 1, action] = player
                return board
        if item == NO_PLAYER and index == board.shape[0] - 1:
            if copy:
                boardCopy = board.copy()
                boardCopy[index, action] = player
                return boardCopy
            else:
                board[index, action] = player
            return board


def connected_four(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """
    diags = get_diags_of_matrix(board)
    winner = find_winner_from_list(diags)

    if winner != NO_PLAYER and winner != player:
        return False
    if winner != NO_PLAYER and winner == player:
        return True
    else:
        winner = find_winner_from_list(board.copy())
        if winner != NO_PLAYER and winner == player:
            return True
        winner = find_winner_from_list(board.copy().T)
        if winner != NO_PLAYER and winner == player:
            return True

    return False


def check_end_state(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    result = connected_four(board, player)
    if result:
        return GameState.IS_WIN

    copyBoard = board.copy()
    if BoardPiece == PLAYER1:
        copyBoard = copyBoard.reshape(6 * 7)
        for index, item in enumerate(copyBoard):
            if item == NO_PLAYER:
                copyBoard[index] = PLAYER2
        copyBoard.reshape(6, 7)
        result = connected_four(copyBoard, PLAYER2)
        if result == False:
            return GameState.IS_DRAW

    if BoardPiece == PLAYER2:
        copyBoard = copyBoard.reshape(6 * 7)
        for index, item in enumerate(copyBoard):
            if item == NO_PLAYER:
                copyBoard[index] = PLAYER1
        copyBoard.reshape(6, 7)
        result = connected_four(copyBoard, PLAYER1)
        if result == False:
            return GameState.IS_DRAW

    return GameState.STILL_PLAYING


def get_diags_of_matrix(board: np.ndarray) -> list:
    a = board.copy()
    # a.diagonal returns the top-left-to-lower-right diagonal "i"
    # according to this diagram:
    #
    #  0  1  2  3  4 ...
    # -1  0  1  2  3
    # -2 -1  0  1  2
    # -3 -2 -1  0  1
    #  :
    #
    # You wanted lower-left-to-upper-right and upper-left-to-lower-right diagonals.
    #
    # The syntax a[slice,slice] returns a new array with elements from the sliced ranges,
    # where "slice" is Python's [start[:stop[:step]] format.

    # "::-1" returns the rows in reverse. ":" returns the columns as is,
    # effectively vertically mirroring the original array so the wanted diagonals are
    # lower-right-to-uppper-left.
    #
    # Then a list comprehension is used to collect all the diagonals.  The range
    # is -x+1 to y (exclusive of y), so for a matrix like the example above
    # (x,y) = (4,5) = -3 to 4.
    diags = [a[::-1, :].diagonal(i) for i in range(-a.shape[0] + 1, a.shape[1])]

    # Now back to the original array to get the upper-left-to-lower-right diagonals,
    # starting from the right, so the range needed for shape (x,y) was y-1 to -x+1 descending.
    diags.extend(a.diagonal(i) for i in range(a.shape[1] - 1, -a.shape[0], -1))

    # Another list comp to convert back to Python lists from numpy arrays,
    # so it prints what you requested.
    result = list(filter(lambda item: len(item) >= 4, [n.tolist() for n in diags]))
    return result


def find_winner_from_list(list):
    winner = NO_PLAYER
    for data in list:
        matchData = [NO_PLAYER, NO_PLAYER]
        matchCount = 0
        for index, player in enumerate(data):
            matchData[0] = player
            if matchData[1] != matchData[0]:
                matchData[1] = player
                matchCount = 1
            elif matchData[1] == matchData[0] and matchData[1] != NO_PLAYER:
                matchCount += 1
                if matchCount == 4:
                    winner = player
                    break
        if winner == NO_PLAYER:
            continue
        else:
            break

    return winner


def can_play(board, column):
    """
    Check if the given column is free
    """
    return board[0, column] == NO_PLAYER


def valid_move(board):
    return [i for i in range(board.shape[1]) if can_play(board, i)]
