import numpy as np
import numpy.testing as nptest
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2

desiredBoardString = "|==============|\n|              |\n|              |\n|    X X       |\n|    O X X     |\n|  O " \
                     "X O " \
                     "O     |\n|  O O X X     |\n|==============|\n|0 1 2 3 4 5 6 |"

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


class TestBoard:

    def test_board_init(self):
        from agents.common import initialize_game_state
        actual = initialize_game_state()
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == BoardPiece
        assert actual.shape == (6, 7)
        assert np.all(actual == NO_PLAYER)

    def test_pretty_board_print(self, capsys):
        from agents.common import pretty_print_board
        ret = pretty_print_board(desiredBoardNp)
        captured = capsys.readouterr()
        assert ret == desiredBoardString

    def test_string_to_board(self):
        from agents.common import string_to_board
        board = string_to_board(desiredBoardString)
        nptest.assert_array_equal(board, desiredBoardNp)
