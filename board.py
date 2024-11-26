from typing import List, Tuple

BOARD = '''
    A
    AA
    AAA
    AAAA
FFFF.....BBBB
 FFF......BBB
  FF.......BB
   F........B
    .........
    E........C
    EE.......CC
    EEE......CCC
    EEEE.....CCCC
         DDDD
          DDD
           DD
            D
'''

BOARD_MAX_DIM = 17
CELL_COUNT = 121
CELL_OOB = -1
XY_TO_CELL = [CELL_OOB] * BOARD_MAX_DIM * BOARD_MAX_DIM
CELL_TO_XY = [()] * CELL_COUNT
A_CELLS = []
D_CELLS = []

cell_no = 0
for y, row in enumerate(BOARD.split('\n')[1:]):
    for x, ch in enumerate(row):
        if ch != ' ':
            CELL_TO_XY[cell_no] = (x, y)
            XY_TO_CELL[y * BOARD_MAX_DIM + x] = cell_no
            cell_no += 1
        if ch == 'A': A_CELLS.append((x, y))
        if ch == 'D': D_CELLS.append((x, y))
del cell_no

def xy_to_cell(x: int, y: int) -> int:
    if x < 0 or x >= BOARD_MAX_DIM or y < 0 or y >= BOARD_MAX_DIM:
        return CELL_OOB
    return XY_TO_CELL[y * BOARD_MAX_DIM + x]

def cell_to_xy(cell: int) -> Tuple[int, int]:
    return CELL_TO_XY[cell]

def list_xy_around(x: int, y: int, size: int) -> List[Tuple[int, int]]:
    if size == 0:
        return []

    result = []

    x0, y0 = x - size, y - size
    stride = size + 1
    k = 0

    while k < size:
        for i in range(stride):
            result.append((x0 + i, y0))
        stride += 1
        k += 1
        y0 += 1

    for i in range(stride):
        if i != size:
            result.append((x0 + i, y0))
    stride -= 1
    y0 += 1

    while k > 0:
        x0 += 1
        for i in range(stride):
            result.append((x0 + i, y0))
        stride -= 1
        k -= 1
        y0 += 1

    return result

def test_list_xy_around():
    xx, yy = 8, 8
    board = [['.'] * 17 for _ in range(17)]
    board[yy][xx] = 'X'
    for (x, y) in list_xy_around(xx, yy, 3):
        board[y][x] = '*'
    out = '\n'.join([''.join(x) for x in board])
    print(out)
    print(len(list_xy_around(xx, yy, 3)))

EMPTY = 0
FRIENDLY = 1
ENEMY = 2

def new_board_state() -> List[int]:
    result = [EMPTY] * 121
    for x, y in A_CELLS:
        cell = xy_to_cell(x, y)
        result[cell] = FRIENDLY
    for x, y in D_CELLS:
        cell = xy_to_cell(x, y)
        result[cell] = ENEMY
    return result

def flip_board_state(state: List[int]) -> List[int]:
    return list(reversed([EMPTY if x == EMPTY else FRIENDLY if x == ENEMY else ENEMY for x in state]))

def print_board(state: List[int]):
    for y in range(BOARD_MAX_DIM):
        for x in range(BOARD_MAX_DIM):
            cell = xy_to_cell(x, y)
            if cell == CELL_OOB:
                print(' ', end='')
            elif state[cell] == FRIENDLY:
                print('X', end='')
            elif state[cell] == ENEMY:
                print('O', end='')
            else:
                print('.', end='')
        print()
