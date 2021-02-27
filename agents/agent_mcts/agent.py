import numpy as np
from agents.agent_mcts.mcts import train_mcts_once
from agents.common import BoardPiece, SavedState, PlayerAction
from typing import Optional, Callable

node = None
mcts = None
trained = False


def get_agent_move(board: np.ndarray, _player: BoardPiece, saved_state: Optional[SavedState]):
    """
    If the game state is already trained using MCTS algorithm, return the best move that the agent can play or
    the agent is made to select its move by applying all the stages of mcts algorithm selection, expansion,
    simulation and back propogation to the set of all the available moves.

    Return the agent move
    """
    global mcts
    global node
    global trained
    if not trained:
        for i in range(100):
            mcts = train_mcts_once(board, mcts)
        print('training finished')

    node = mcts
    #new_node, move = node.select_move()
    # node = train_mcts_during(node, training_time)
    # print([(n.win, n.games) for n in node.children])
    node, move = node.select_move()

    return PlayerAction(move), saved_state
