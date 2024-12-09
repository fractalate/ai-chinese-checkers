from typing import List, Tuple, Union

from collections import deque
import functools

import torch

functools.cache
BOARD = '''
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
'''

BOARD_GRID_DIM = 17
OOB = 0
NOBODY = -1

# "new" is a lie in this function name.
@functools.cache
def _new_board_state_tensor(num_players: int) -> torch.Tensor:
    state = torch.empty((BOARD_GRID_DIM, BOARD_GRID_DIM), dtype=torch.int32)
    for row, row_text in enumerate(BOARD.rstrip().split('\n')[1:]):
        col = 0
        while col < len(row_text) and col < BOARD_GRID_DIM:
            c = row_text[col]
            if c == ' ':
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
    return _new_board_state_tensor(num_players).clone()

def get_opposition_player_no(player_no: int) -> int:
    return player_no - 1 if player_no % 2 == 0 else player_no + 1

# End zones are across eachother.
BOARD_6 = new_board_state_tensor(6)
def is_valid_cell_for_player(row: int, col: int, player_no: int) -> bool:
    if row < 0 or row >= BOARD_GRID_DIM or col < 0 or col >= BOARD_GRID_DIM:  # TODO Do I want this guard here or elsewhere?
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

'''
##
#X#
 ##
'''
ADJACENT_CELLS = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)]

'''
# #
 ..
#.X.#
  ..
  # #
'''
ADJACENT_CELLS_2 = [(-2, -2), (-2, 0), (0, -2), (0, 2), (2, 0), (2, 2)]


class Board:
    def __init__(self, skip_init: bool = False):
        '''
        :param skip_init: If set to True, the state tensor will not be initialized with zeros (for efficiency).
        '''
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

    def is_valid_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
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
            return True
        # Do a bredth first search of valid hops to assess multi-hops.
        queue = deque([(from_row, from_col)])
        backtracking = set(queue)
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
                elif tr == to_row and tc == to_col:
                    return True  # If we arrive at our "to" cell, the sequence of hops was valid.
                # Otherwise, the hop might be valid, so consider it later, but only if the hop is not a backtrack.
                entry = (tr, tc)
                if entry not in backtracking:
                    queue.append(entry)
                    backtracking.add(entry)
        return False

    def dumps(self, highlight_cells: List[Tuple[int, int]] = None) -> str:
        result = []
        for row in range(BOARD_GRID_DIM):
            line = []
            for col in range(BOARD_GRID_DIM):
                if highlight_cells and (row, col) in highlight_cells:
                    line.append('*')
                elif self.state[row, col] == OOB:
                    line.append(' ')
                elif self.state[row, col] == NOBODY:
                    line.append('.')
                else:
                    line.append(str(self.state[row, col].item()))
            result.append(''.join(line))
        return '\n'.join(result)

b = Board()
b.setup_board(2)
b.state[5, 4] = 1
b.state[7, 4] = 1
r0, c0 = 2, 4
highlight_cells = []
for r in range(BOARD_GRID_DIM):
    for c in range(BOARD_GRID_DIM):
        if b.is_valid_move(r0, c0, r, c):
            highlight_cells.append((r, c))
print(b.dumps(highlight_cells=highlight_cells))
