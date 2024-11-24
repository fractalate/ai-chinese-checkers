from typing import List
import board

EMPTY = 0
FRIENDLY = 1
ENEMY = 2

def new_board_state() -> List[int]:
    result = [EMPTY] * 121
    for x, y in board.A_CELLS:
        cell = board.xy_to_cell(x, y)
        result[cell] = FRIENDLY
    for x, y in board.D_CELLS:
        cell = board.xy_to_cell(x, y)
        result[cell] = ENEMY
    return result

def flip_board_state(state: List[int]) -> List[int]:
    return list(reversed(EMPTY if x == EMPTY else FRIENDLY if x == ENEMY else ENEMY for x in state))

