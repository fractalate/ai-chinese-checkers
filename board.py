from typing import Tuple

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

XY_TO_CELL = [-1] * 17 * 17
CELL_TO_XY = [-1] * 121

cell_no = 0
for y, row in enumerate(BOARD.split('\n')[1:]):
    for x, ch in enumerate(row):
        if ch != ' ':
            CELL_TO_XY[cell_no] = (x, y)
            XY_TO_CELL[y * 17 + x] = cell_no
            cell_no += 1
del cell_no

def xy_to_cell(x: int, y: int) -> int:
    return XY_TO_CELL[y * 17 + x]

def cell_to_xy(cell: int) -> Tuple[int, int]:
    return CELL_TO_XY[cell]
