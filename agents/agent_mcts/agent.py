import numpy as np
from agents.agent_mcts.mcts import train_mcts_once
from agents.common import BoardPiece, SavedState, PlayerAction
from typing import Optional, Callable

node = None
mcts = None
trained = False


def get_agent_move(board: np.ndarray, _player: BoardPiece, saved_state: Optional[SavedState]):
   
    """
    This function returns the mcts agent move.

    When the game is starting, and when this function is called for the first time, the algorithm will be trained
    for the certain amount of iterations.
    
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
