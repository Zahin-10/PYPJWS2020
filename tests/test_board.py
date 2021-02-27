import numpy as np
import numpy.testing as nptest
import sys, os

from agents.agent_mcts.mcts import train_mcts_once
from agents.agent_mcts import Node
from agents.agents_minimax.minimax import can_play_spot
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2, PlayerAction, initialize_game_state, \
    pretty_print_board, string_to_board, apply_player_action, connected_four, get_player_to_play, valid_move

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

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
        actual = initialize_game_state()
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == BoardPiece
        assert actual.shape == (6, 7)
        assert np.all(actual == NO_PLAYER)

    def test_pretty_board_print(self, capsys):
        ret = pretty_print_board(desiredBoardNp)
        assert ret == desiredBoardString

    def test_string_to_board(self):
        board = string_to_board(desiredBoardString)
        nptest.assert_array_equal(board, desiredBoardNp)

    def test_apply_player_action(self):
        desiredBoardNp[0, 4] = PLAYER1
        desiredBoardNp[1, 4] = PLAYER2
        desiredBoardNp[2, 4] = PLAYER1
        copyBoard = desiredBoardNp.copy()
        copyBoard[3, 1] = PLAYER1
        modifiedBoard = apply_player_action(desiredBoardNp, PlayerAction(1), PLAYER1)
        assert np.array_equal(desiredBoardNp, copyBoard)
        assert np.array_equal(desiredBoardNp, modifiedBoard)
        modifiedBoard = apply_player_action(desiredBoardNp, PlayerAction(1), PLAYER1, True)
        assert np.array_equal(desiredBoardNp, modifiedBoard) == False
        newBoard = apply_player_action(desiredBoardNp, PlayerAction(4), PLAYER2)
        assert newBoard == None

    def test_connected_four(self):
        # Check column
        copyBoardForColTest = desiredBoardNp.copy()
        copyBoardForColTest[4, 3] = PLAYER1
        copyBoardForColTest[1, 3] = PLAYER1
        assert connected_four(copyBoardForColTest, PLAYER1) == True
        assert connected_four(copyBoardForColTest, PLAYER2) == False
        # Check diagonal
        copyBoardForDiagTest = desiredBoardNp.copy()
        copyBoardForDiagTest[5, 1] = PLAYER1
        copyBoardForDiagTest[2, 4] = PLAYER1
        assert connected_four(copyBoardForColTest, PLAYER1) == True
        assert connected_four(copyBoardForColTest, PLAYER2) == False
        # Check row
        copyBoardForRowTest = desiredBoardNp.copy()
        copyBoardForRowTest[5, 1] = PLAYER1
        copyBoardForRowTest[5, 2] = PLAYER1
        assert connected_four(copyBoardForRowTest, PLAYER1) == True
        assert connected_four(copyBoardForRowTest, PLAYER2) == False

    def test_get_player_to_play(self):
        print(get_player_to_play(desiredBoardNp))

    def test_mcts_select_move(self):
        mcts = None
        for i in range(100):
            mcts = train_mcts_once(desiredBoardNp, mcts)
        node = mcts
        valid_moves = valid_move(node.state)
        node, move = node.select_move()
        assert isinstance(node, Node)
        mask = np.array(valid_moves) == move
        assert np.any(mask)

    def test_can_play_spot(self):
        pos = (1, 2)
        spot_check_board = desiredBoardNp.copy()
        assert can_play_spot(pos, spot_check_board) == True
        pos = (0, 2)
        assert can_play_spot(pos, spot_check_board) == False
