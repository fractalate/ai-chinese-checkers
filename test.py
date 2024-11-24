from typing import List
import torch
import torch.nn as nn

import board
import game

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# l is size of neighborhood around pieces to consider
def calculate_input_neurons(l: int) -> int:
    # For each of the 121 cells on the board:
    # - one input if cell is friendly
    # - one input if cell is enemy
    # - one input if cell is backtracking
    # For each of the 20 pieces in a 2 player game:
    #   For each of the COUNT cells around each piece
    #   - one input if cell is friendly
    #   - one input if cell is enemy
    #   - one input if cell is impassible
    #   - one input if cell is backtracking
    #
    # Layout:
    #   [friendly] * 121
    #   [enemy] * 121
    #   [backtracking] * 121
    #   [
    #     [
    #       friendly
    #       enemy
    #       impassable
    #       backtracking
    #     ] * COUNT
    #   ] * 20
    COUNT = len(board.list_xy_around(0, 0, l))
    return 121 * 3 + 4 * 20 * COUNT

def board_state_to_input_vector(state: List[int], backtracking_cells: List[int] = []) -> List[int]:
    result = []
    # board state
    result += [1.0 if p == board.FRIENDLY else 0.0 for p in state]
    result += [1.0 if p == board.ENEMY else 0.0 for p in state]
    result += [1.0 if cell in backtracking_cells else 0.0 for cell in range(len(state))]
    # local-to-piece state
    for cell, pawn in enumerate(state):
        if pawn != board.FRIENDLY and pawn != board.ENEMY:
            continue
        x, y = board.cell_to_xy(cell)
        for adj_x, adj_y in board.list_xy_around(x, y, L):
            adj_cell = board.xy_to_cell(adj_x, adj_y)
            adj_pawn = state[adj_cell] if adj_cell != board.CELL_OOB else -1
            result += [
                1.0 if adj_pawn == board.FRIENDLY else 0.0,
                1.0 if adj_pawn == board.ENEMY else 0.0,
                1.0 if adj_cell == board.CELL_OOB else 0.0,
                1.0 if adj_cell in backtracking_cells else 0.0,
            ]
    assert len(result) == N
    return torch.tensor(result)

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

L = 6 # is this too big?
N = calculate_input_neurons(L)
H = 100_000 # how big is too big?
M = 121*121 + 1
print(f'{L=} {N=} {H=} {M=}')

model1 = NeuralNetwork(input_size=N, output_size=M, hidden_size=H)
model2 = NeuralNetwork(input_size=N, output_size=M, hidden_size=H)

state = board.new_board_state()

for i in range(100):
    model = model1

    inputs = board_state_to_input_vector(state)
    outputs = model(inputs)

    best_action = choose_best_action(state, outputs)
    if best_action is None:
        print()
        print('Best Action: Do Nothing')
    else:
        fc, tc = best_action
        fx, fy = board.cell_to_xy(fc)
        tx, ty = board.cell_to_xy(tc)
        print()
        print(f'Best Action: ({fx}, {fy}) to ({tx}, {ty})')

        state[tc] = state[fc]
        state[fc] = board.EMPTY

    board.print_board(state if i % 2 == 0 else board.flip_board_state(state))
    state = board.flip_board_state(state)
    model1, model2 = model2, model1
