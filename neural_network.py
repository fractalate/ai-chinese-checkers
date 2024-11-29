from typing import List, Tuple, Union

import torch
import torch.nn as nn

from game import Game
from board import (
    NOBODY,
    CELL_OOB,
    CELL_COUNT,
    cell_to_xy,
    list_xy_around,
    xy_to_cell,
)

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
    COUNT = len(list_xy_around(0, 0, l))
    return 121 * 3 + 4 * 20 * COUNT

# Returns (cells, backtracking_cells) for the models perspective.
def objective_pov_to_model_pov(game: Game, player: int, backtracking_cells: Union[None, List[int]]) -> Tuple[ List[int], List[int] ]:
    if player == 1:
        return (game.board.cells, backtracking_cells if backtracking_cells else [])
    elif player == 2:
        return (
            list(reversed(game.board.cells)), # TODO Gross
            list(model_pov_to_objective_pov(player, x) for x in (backtracking_cells if backtracking_cells else [])),
        )
    raise NotImplementedError()

# Return cell.
def model_pov_to_objective_pov(player: int, cell: int) -> int:
    if player == 1:
        return cell
    elif player == 2:
        return CELL_COUNT - cell - 1
    raise NotImplementedError()

def game_state_to_input_vector(game: Game, player: int, backtracking_cells: Union[None, List[int]] = None) -> List[int]:
    cells, backtracking_cells = objective_pov_to_model_pov(game, player, backtracking_cells)

    result = []

    # board state
    result += [1.0 if pawn == player else 0.0 for pawn in cells]
    result += [1.0 if pawn != player and pawn != NOBODY else 0.0 for pawn in cells]
    result += [1.0 if cell in backtracking_cells else 0.0 for cell in range(len(cells))]

    # local-to-piece state
    for cell, pawn in enumerate(cells):
        if pawn == NOBODY:
            continue
        x, y = cell_to_xy(cell)
        for adj_x, adj_y in list_xy_around(x, y, L):
            adj_cell = xy_to_cell(adj_x, adj_y)
            adj_pawn = cells[adj_cell] if adj_cell != CELL_OOB else -1
            result += [
                1.0 if adj_pawn == player else 0.0,
                1.0 if adj_pawn != player and adj_pawn != NOBODY else 0.0,
                1.0 if adj_cell == CELL_OOB else 0.0,
                1.0 if adj_cell in backtracking_cells else 0.0,
            ]

    assert len(result) == N

    # TODO Maybe do it all in the tensor from the start.
    return torch.tensor(result)

# Returns (from_cell, to_cell).
def choose_best_action(game: Game, player: int, outputs: torch.Tensor) -> Union[None, Tuple[int, int]]:
    # otherwise we have to check whether some moves are valid
    best_action_score = -100.0
    best_action = None
    for out, action in enumerate(outputs):
        score = action.item()
        _to_cell = (out // 121)
        if _to_cell == 121:  # no action case
            if score > best_action_score:
                if game.is_valid_no_action():
                    best_action = None
                    best_action_score = score
        else:
            from_cell = model_pov_to_objective_pov(player, out % 121)
            to_cell = model_pov_to_objective_pov(player, _to_cell)
            if score > best_action_score:
                if game.is_valid_move(from_cell, to_cell):
                    best_action = (from_cell, to_cell)
                    best_action_score = score
    assert best_action_score > -100.0  # todo is it possible for these scores to dip real low, like lower than this? If this  is -100 exactly, probably forgot to change pov
    # todo - this has to be amended to be recursive so we can do multi-hops
    return best_action

L = 4  # is this too big?
N = calculate_input_neurons(L)
H = 10_000  # how big is too big?
M = 121*121 + 1

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(N, H)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(H, M)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
