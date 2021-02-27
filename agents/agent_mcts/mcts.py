import numpy as np
import random

from agents.agent_mcts import Node
from agents.common import initialize_game_state, valid_move, PLAYER1, can_play, apply_player_action, check_end_state, \
    GameState, PLAYER2, get_player_to_play


def play(board_, column, player=None):
    """
    Play at given column, if no player provided, calculate which player must play, otherwise force player to play
    Return new board and winner
    """
    board = board_.copy()
    if player is None:
        player = get_player_to_play(board)
    if can_play(board, column):
        board = apply_player_action(board, column, player)
    else:
        raise Exception('Error : Column {} is full'.format(column))
    return board, player if check_end_state(board, player) == GameState.IS_WIN else 0


def random_play_improved(grid):
    """
    When UCB can no longer be applied to find the next node, the game tree is expanded further to include an unexplored
    child randomly until it reaches the final stage of the game
    """
    def get_winning_moves(grid, moves, player):
        return [move for move in moves if play(grid, move, player=player)[1]]

    # If can win, win
    while True:
        moves = valid_move(grid)
        if len(moves) == 0:
            return 0
        player_to_play = get_player_to_play(grid)
        opponent = PLAYER2 if player_to_play == PLAYER1 else PLAYER1
        winning_moves = get_winning_moves(grid, moves, player_to_play)
        loosing_moves = get_winning_moves(grid, moves, opponent)

        if len(winning_moves) > 0:
            selected_move = winning_moves[0]
        elif len(loosing_moves) == 1:
            selected_move = loosing_moves[0]
        else:
            selected_move = random.choice(moves)
        grid, winner = play(grid, selected_move)
        if np.abs(winner) > 0:
            return player_to_play


def train_mcts_during(mcts, training_time):
    import time
    start = int(round(time.time() * 1000))
    current = start
    while (current - start) < training_time:
        mcts = train_mcts_once(mcts)
        current = int(round(time.time() * 1000))
    return mcts


def train_mcts_once(init_board, mcts=None):
    if mcts is None:
        mcts = Node(init_board, 0, None, None)

    node = mcts

    # selection
    while node.children is not None:
        # Select highest uct
        ucts = [child.get_uct() for child in node.children]
        if None in ucts:
            node = random.choice(node.children)
        else:
            node = node.children[np.argmax(ucts)]

    # expansion, no expansion if terminal node
    moves = valid_move(node.state)
    if len(moves) > 0:

        if node.winner == 0:

            states = [(play(node.state, move), move) for move in moves]
            node.set_children(
                [Node(state_winning[0], state_winning[1], move=move, parent=node) for state_winning, move in states])
            # simulation
            winner_nodes = [n for n in node.children if n.winner]
            if len(winner_nodes) > 0:
                node = winner_nodes[0]
                victorious = node.winner
            else:
                node = random.choice(node.children)
                victorious = random_play_improved(node.state)
        else:
            victorious = node.winner

        # backpropagation
        parent = node
        while parent is not None:
            parent.games += 1
            if victorious != 0 and get_player_to_play(parent.state) != victorious:
                parent.win += 1
            parent = parent.parent


    else:
        print('no valid moves, expended all')

    return mcts
