from enum import Enum
from typing import Optional
import numpy as np

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

PlayerAction = np.int8  # The column to be played


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
                boardCol = int(colindex/2)
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
    raise NotImplementedError()


def connected_four(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """
    raise NotImplementedError()


def check_end_state(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    raise NotImplementedError()
