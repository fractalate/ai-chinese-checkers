import os
from typing import List

import torch
import torch.nn as nn

from neural_network import NeuralNetwork
import board

class Player:
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.moves = 0
        self.winner = False
        self.loser = False

    def score(self, state: List[int]) -> float:
        total = 0.0
        if self.winner:
            total += 1000.0
        if self.loser:
            total -= 1000.0
        for cell, pawn in enumerate(state):
            x, y = board.cell_to_xy(cell)
            if pawn == board.FRIENDLY:
                total += y
            if cell in board.D_CELLS:
                total += 10.0
        return total

def is_valid_move(state: List[int], from_cell: int, to_cell: int) -> bool:
    fx, fy = board.cell_to_xy(from_cell)
    tx, ty = board.cell_to_xy(to_cell)
    if state[from_cell] != board.FRIENDLY:
        return False
    arounds_delta = board.list_xy_around(0, 0, 1)
    arounds_delta2 = [(x*2, y*2) for (x, y) in arounds_delta]
    if (tx-fx, ty-fy) in arounds_delta:
        if state[to_cell] in [board.FRIENDLY, board.ENEMY]:
            return False
        return True
    elif (tx-fx, ty-fy) in arounds_delta2:
        i = arounds_delta2.index((tx-fx, ty-fy))
        dhx, dhy = arounds_delta[i]
        hx, hy = fx+dhx, fy+dhy
        hcell = board.xy_to_cell(hx, hy)
        assert hcell >= 0 and hcell < 121
        if state[hcell] not in [board.FRIENDLY, board.ENEMY]:
            return False
        if state[to_cell] in [board.FRIENDLY, board.ENEMY]:
            return False
        return True
    return False

def is_game_won_by_friendly(state: List[int]) -> bool:
    count_filled = 0
    has_friendly_pawn = False
    for (x, y) in board.D_CELLS:
        cell = board.xy_to_cell(x, y)
        if state[cell] == board.FRIENDLY:
            has_friendly_pawn = True
        if state[cell] != board.EMPTY:
            count_filled += 1
    return has_friendly_pawn and len(board.D_CELLS) == count_filled

def is_game_lost_by_friendly(state: List[int]) -> bool:
    state = board.flip_board_state(state)
    return is_game_won_by_friendly(state)

def is_valid_no_action(state: List[int]) -> bool:
    return False # todo - only possible to do nothing when in a multi-hop

def choose_best_action(state: List[int], outputs):
    # otherwise we have to check whether some moves are valid
    best_action_score = -100.0
    best_action = None
    for out, action in enumerate(outputs):
        score = action.item()
        from_cell = out % 121
        to_cell = (out // 121)
        if to_cell == 121: # no action case
            if score > best_action_score:
                if is_valid_no_action(state):
                    best_action = None
                    best_action_score = score
        else:
            if score > best_action_score:
                if is_valid_move(state, from_cell, to_cell):
                    best_action = (from_cell, to_cell)
                    best_action_score = score
    assert best_action_score > -100.0 # todo is it possible for these scores to dip real low, like lower than this?
    # todo - this has to be amended to be recursive so we can do multi-hops
    return best_action

# Thoughts About Scoring
# 
# * Does it make sense for an AI to prioritize clearing out its starting zone?
#   - I think so! Because it makes more space for the opponent to get their pieces in.

def train():
    try:
        os.mkdir('generations')
    except FileExistsError:
        pass
    max_gen_no = 1
    for gen in os.listdir('generations'):
        gen_no = int(gen)
        if gen_no > max_gen_no:
            max_gen_no = gen_no
        del gen_no
    gen_path = os.path.join('generations', str(max_gen_no))
    try:
        os.mkdir(gen_path)
    except FileExistsError:
        pass
    train_generation(max_gen_no)

def get_models_for_generation(gen: int):
    model_nos = []
    gen_path = os.path.join('generations', str(gen))
    for model in os.listdir(gen_path):
        if model.endswith('.pth'):
            model_no = int(model[:-4])
            model_nos.append(model_no)
    return model_nos

def train_generation(gen: int):
    model_nos = get_models_for_generation(gen)
    if gen == 1:
        new_model_no = 1
        while len(model_nos) < 10:
            while new_model_no in model_nos:
                new_model_no += 1
            model_nos.append(new_model_no)
            print(f'Creating model {new_model_no} for generation {gen}...')
            model = NeuralNetwork()
            where = os.path.join('generations', str(gen), str(new_model_no) + '.pth')
            torch.save(model.state_dict(), where)
            print(f'Model saved to {where}')

def play_a_match():
    player1 = Player(
        model=NeuralNetwork(),
        name='Player 1',
    )

    player2 = Player(
        model=NeuralNetwork(),
        name='Player 2',
    )

    state = board.new_board_state()
    winner = None

    for i in range(30):
        if is_game_won_by_friendly(state):
            winner = player1
            player1.winner = True
            player2.loser = True
            break

        xplayer = player1 if i % 2 == 0 else player2
        oplayer = player2 if i % 2 == 0 else player1

        inputs = board_state_to_input_vector(state)
        outputs = player1.model(inputs)

        print()
        print(f'X - {xplayer.name}')
        print(f'O - {oplayer.name}')

        best_action = choose_best_action(state, outputs)
        if best_action is None:
            print('Best Action: Do Nothing')
        else:
            fc, tc = best_action
            fx, fy = board.cell_to_xy(fc)
            tx, ty = board.cell_to_xy(tc)
            print(f'Best Action: ({fx}, {fy}) to ({tx}, {ty})')

            state[tc] = state[fc]
            state[fc] = board.EMPTY
            player1.moves += 1

        if is_game_won_by_friendly(state):
            winner = player1
            player1.winner = True
            player2.loser = True
            break

        board.print_board(state if i % 2 == 0 else board.flip_board_state(state))
        state = board.flip_board_state(state)
        player1, player2 = player2, player1

    if winner is None:
        print('The only way to win is to not play the game. These models did not "play". (Draw)')
    else:
        print(f'Winner: {winner}!')

    p1 = player1.score(state)
    p2 = player2.score(board.flip_board_state(state))

    print(f'{player1.name} - Score: {p1}')
    print(f'{player2.name} - Score: {p2}')

if __name__ == '__main__':
    train()
