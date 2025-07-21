from typing import Generator, List, Tuple, Union

from collections import deque
import functools

import torch

BOARD = """
    1
    11
    111
    1111
6666.....3333
 666......333
  66.......33
   6........3
    .........
    4........5
    44.......55
    444......555
    4444.....5555
         2222
          222
           22
            2
"""

"""
##
#X#
 ##
"""
ADJACENT_CELLS = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)]

"""
# #
 ..
#.X.#
  ..
  # #
"""
ADJACENT_CELLS_2 = [(-2, -2), (-2, 0), (0, -2), (0, 2), (2, 0), (2, 2)]

BOARD_GRID_DIM = 17
OOB = 0
NOBODY = -1


@functools.cache
def _lookup_board_state_tensor(num_players: int) -> torch.Tensor:
    state = torch.empty((BOARD_GRID_DIM, BOARD_GRID_DIM), dtype=torch.int32)
    for row, row_text in enumerate(BOARD.rstrip().split("\n")[1:]):
        col = 0
        while col < len(row_text) and col < BOARD_GRID_DIM:
            c = row_text[col]
            if c == " ":
                state[row, col] = OOB
            elif not c.isdigit():
                state[row, col] = NOBODY
            elif (player_no := int(c)) <= num_players:
                state[row, col] = player_no
            else:
                state[row, col] = NOBODY
            col += 1
        while col < BOARD_GRID_DIM:
            state[row, col] = OOB
            col += 1
    return state


def new_board_state_tensor(num_players: int) -> torch.Tensor:
    return _lookup_board_state_tensor(num_players).clone()


BOARD_6 = new_board_state_tensor(6)


def get_opposition_player_no(player_no: int) -> int:
    return player_no - 1 if player_no % 2 == 0 else player_no + 1


# End zones are across from each other.
def is_valid_cell_for_player(row: int, col: int, player_no: int) -> bool:
    if (
        row < 0 or row >= BOARD_GRID_DIM or col < 0 or col >= BOARD_GRID_DIM
    ):  # TODO Do I want this guard here or elsewhere?
        return False
    cell_value = BOARD_6[row, col]
    if cell_value == OOB:
        return False  # Can't go OOB.
    elif cell_value == NOBODY:
        return True  # Can go into any empty cell in template board.
    elif cell_value == player_no:
        return True  # Can go into player's own end-zone.
    # It must be another player's end-zone, so the cell is valid only if it's for the opposition.
    return cell_value == get_opposition_player_no(player_no)


@functools.cache
def get_goal_cells(player_no: int) -> List[int]:
    opposition = get_opposition_player_no(player_no)
    results = []
    for row in range(BOARD_GRID_DIM):
        for col in range(BOARD_GRID_DIM):
            if BOARD_6[row, col] == opposition:
                results.append((row, col))
    return results


class Board:
    GRID_DIM = BOARD_GRID_DIM

    def __init__(self, skip_init: bool = False):
        """
        :param skip_init: If set to True, the state tensor will not be initialized with zeros (for efficiency).
        """
        # The board is represented by a 17x17 grid filled with integer values.
        # OOB (0) means out of bounds.
        # NOBODY (-1) means unoccupied space.
        # Any positive number means a piece belonging to the player with that number.
        # Index like self.state[row, column]; zero indexed.
        self.state: torch.Tensor = None
        if not skip_init:
            self.state: torch.Tensor = new_board_state_tensor(0)

    def setup_board(self, num_players: int):
        self.state = new_board_state_tensor(num_players)

    def is_valid_move(
        self,
        from_row: int,
        from_col: int,
        to_row: int,
        to_col: int,
        out_moves: None | List[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> bool:
        # Gotta move.
        if from_row == to_row and from_col == to_col:
            return False
        # Check that the "from" cell is some player's piece.
        player_no = self.state[from_row, from_col]
        if player_no <= 0:
            return False
        # Check that the "to" cell is unoccupied.
        target = self.state[to_row, to_col]
        if target != NOBODY:
            return False
        # Check that the "to" cell is one of the player's valid spaces.
        if not is_valid_cell_for_player(to_row, to_col, player_no):
            return False
        # Consider single jumps.
        if (to_row - from_row, to_col - from_col) in ADJACENT_CELLS:
            if out_moves is not None:
                out_moves.append((from_row, from_col), (to_row, to_col))
            return True
        if from_row % 2 != to_row % 2 or from_col % 2 != to_col % 2:
            return False  # A hop always preserves odd/even-ness of the row and column.
        # Do a breadth first search of valid hops to assess multi-hops.
        queue = deque([(from_row, from_col)])
        backtracking = set(queue)
        if out_moves is not None:
            parent_cell = dict()  # XXX We're not tracking the minimum cost paths, but that might be nicer to do.
        while queue:
            fr, fc = queue.popleft()
            for dr, dc in ADJACENT_CELLS_2:
                tr = fr + dr
                tc = fc + dc
                if tr < 0 or tr >= BOARD_GRID_DIM or tc < 0 or tc >= BOARD_GRID_DIM:
                    continue  # Player must stay on the board.
                elif self.state[tr, tc] != NOBODY:
                    continue  # Hop must be to an unoccupied cell.
                # Find the cell that's hopped over.
                tr1 = fr + dr // 2
                tc1 = fc + dc // 2
                if self.state[tr1, tc1] <= 0:
                    continue  # Player must hop a piece.
                entry = (tr, tc)
                if out_moves is not None:
                    parent_cell[entry] = (fr, fc)
                # If we arrive at our "to" cell, the sequence of hops was valid.
                if tr == to_row and tc == to_col:
                    if out_moves is not None:
                        while parent := parent_cell.get(entry):
                            out_moves.append((parent, entry))
                            entry = parent
                        out_moves.reverse()
                    return True
                # Otherwise, the hop might be valid, so consider it later, but only if the hop is not a backtrack.
                if entry not in backtracking:
                    queue.append(entry)
                    backtracking.add(entry)
        return False

    def move(self, from_row: int, from_col: int, to_row: int, to_col: int):
        self.state[to_row, to_col] = self.state[from_row, from_col]
        self.state[from_row, from_col] = NOBODY

    def dumps(self, highlight_cells: List[Tuple[int, int]] = None) -> str:
        result = []
        for row in range(BOARD_GRID_DIM):
            line = []
            for col in range(BOARD_GRID_DIM):
                if highlight_cells and (row, col) in highlight_cells:
                    line.append("*")
                elif self.state[row, col] == OOB:
                    line.append(" ")
                elif self.state[row, col] == NOBODY:
                    line.append(".")
                else:
                    line.append(str(self.state[row, col].item()))
            result.append("".join(line))
        return "\n".join(result)
