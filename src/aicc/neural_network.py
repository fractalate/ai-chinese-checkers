from typing import Optional, Tuple
import torch
import torch.nn as nn

from aicc.game import Game
from aicc.board import Board

# There are 121 cells for pawns in the game of Chinese Checkers.
#
# We restrict ourselves to two players in this implementation.
# We restrict the game to being oriented with the neural network's home zone near them (so the board may need to be reversed).
#
# We'll encode the state of the game into a format suitable for input into the neural network.
# Input consists of two values for each cell on the board encoded in two segments.
# First, the occupation segment, which is 121 values having 0 for unoccupied and 1 for occupied.
# Second, the foe segment, which is 121 values having 0 for not a foe and 1 for is a foe.
#
# Output of the model consists of a 121 x 121 matrix of possible moves.
# The entry with the highest value has a particular row and column.
# The row is the "from" cell for the move.
# The column is the "to" cell for the move.
# The model may produce invalid moves, but the valid move with the highest score is effectively the move the model has chosen.


NUMBER_OF_CELLS = 121

NUMBER_OF_INPUT_NEURONS = NUMBER_OF_CELLS * 2
NUMBER_OF_HIDDEN_NEURONS = 10_000
NUMBER_OF_OUTPUT_NEURONS = NUMBER_OF_CELLS * NUMBER_OF_CELLS

def create_cell_maps():
    game = Game(num_players=2)
    cell_by_row_col = {}
    row_col_by_cell = []

    indexes = range(Board.GRID_DIM)

    for row in indexes:
        for col in indexes:
            cell = game.board.state[row, col]
            if cell == Board.OOB:
                continue
            cell_by_row_col[(row, col)] = len(row_col_by_cell)
            row_col_by_cell.append((row, col))

    return cell_by_row_col, row_col_by_cell


CELL_BY_ROW_COL, ROW_COL_BY_CELL = create_cell_maps()


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(NUMBER_OF_INPUT_NEURONS, NUMBER_OF_HIDDEN_NEURONS, dtype=torch.float64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(NUMBER_OF_HIDDEN_NEURONS, NUMBER_OF_OUTPUT_NEURONS, dtype=torch.float64)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def game_state_to_input_vector(game: Game, player_no: int):
    input_vector = torch.empty(NUMBER_OF_INPUT_NEURONS, dtype=torch.float64)
    cell_no = 0

    # Player 0's orientation is the natural orientation.
    flip = player_no != 1
    # But if it's not this player's turn, flip the board so it considers the board as the other player would.
    if game.player_up != player_no:
        flip = not flip

    indexes = range(Board.GRID_DIM - 1, -1, -1) if flip else range(Board.GRID_DIM)

    for row in indexes:
        for col in indexes:
            cell = game.board.state[row, col]
            if cell == Board.OOB:
                continue

            if cell != Board.NOBODY:
                # Occupation segment.
                input_vector[cell_no] = 1.0
                if cell != player_no:
                    # Foe segment.
                    input_vector[NUMBER_OF_CELLS + cell_no] = 1.0

            cell_no += 1

    return input_vector


def model_pov_to_objective_pov(player_no: int, cell: int) -> int:
    if player_no == 1:
        return cell
    elif player_no == 2:
        return NUMBER_OF_CELLS - cell - 1
    raise NotImplementedError()


def choose_best_move(model: NeuralNetwork, game: Game) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    input_vector = game_state_to_input_vector(game, game.player_up)

    actions = model(input_vector)
    best_action = None
    best_action_score = float("-inf")
    for out, action in enumerate(actions):
        score = action.item()
        _to_cell = (out // 121)
        if score > best_action_score:
            from_cell = model_pov_to_objective_pov(game.player_up, out % 121)
            from_row, from_col = from_row_col = ROW_COL_BY_CELL[from_cell]
            to_cell = model_pov_to_objective_pov(game.player_up, _to_cell)
            to_row, to_col = to_row_col = ROW_COL_BY_CELL[to_cell]
            if game.is_valid_move(from_row, from_col, to_row, to_col):
                best_action = (from_row_col, to_row_col)
                best_action_score = score

    return best_action


if __name__ == "__main__":
    model = NeuralNetwork()
    game = Game(2)

    move = choose_best_move(model, game)
    print("Best Move:", move)
    if move:
        game.move(*move[0], *move[1])
        print(game.board.dumps())
    print()

    move = choose_best_move(model, game)
    print("Best Move:", move)
    if move:
        game.move(*move[0], *move[1])
        print(game.board.dumps())
    print()

    import os
    os.makedirs("out", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("out", "sample.model"))
