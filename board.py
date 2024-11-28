from typing import List, Tuple, Union

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
CELL_TO_XY = [(0, 0)] * CELL_COUNT
XY_TO_CELL = [CELL_OOB] * BOARD_MAX_DIM * BOARD_MAX_DIM

A_CELLS = []
D_CELLS = []

NOBODY = 0

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

"""
def test_list_xy_around():
    xx, yy = 8, 8
    board = [['.'] * 17 for _ in range(17)]
    board[yy][xx] = 'X'
    for (x, y) in list_xy_around(xx, yy, 3):
        board[y][x] = '*'
    out = '\n'.join([''.join(x) for x in board])
    print(out)
    print(len(list_xy_around(xx, yy, 3)))
"""

class Board:
    def __init__(self):
        self.cells: List[int] = [NOBODY] * CELL_COUNT
        self.setup_player(1)
        self.setup_player(2)

    def setup_player(self, player: int):
        for cell in self.get_player_start_cells(player):
            self.cells[cell] = player

    def place_pawn(self, player: int, cell: int):
        self.cells[cell] = player

    def clear_cell(self, cell: int):
        self.cells[cell] = NOBODY

    def get_player_start_cells(self, player: int):
        if player == 1:
            return list(range(10))
        if player == 2:
            return list(range(121 - 10, 121))
        raise NotImplementedError()

    def is_valid_move(self, from_cell: int, to_cell: int, backtracking_cells: Union[None, List[int]]) -> bool:
        if backtracking_cells and to_cell in backtracking_cells:
            return False
        if self.cells[from_cell] == NOBODY:
            return False
        if self.cells[to_cell] != NOBODY:
            return False
        fx, fy = cell_to_xy(from_cell)
        tx, ty = cell_to_xy(to_cell)
        arounds_delta = list_xy_around(0, 0, 1)
        if (tx-fx, ty-fy) in arounds_delta:
            return True
        arounds_delta2 = [(x*2, y*2) for (x, y) in arounds_delta]
        if (tx-fx, ty-fy) in arounds_delta2:
            i = arounds_delta2.index((tx-fx, ty-fy))
            dhx, dhy = arounds_delta[i]
            hx, hy = fx+dhx, fy+dhy
            hopped_cell = xy_to_cell(hx, hy)
            assert hopped_cell != CELL_OOB
            return self.cells[hopped_cell] != NOBODY
        # TODO - No entry into other territory
        return False

    def move(self, from_cell: int, to_cell: int):
        self.cells[to_cell] = self.cells[from_cell]
        self.cells[from_cell] = NOBODY

    def print(self):
        print(self.dumps())

    def dumps(self):
        result = ''
        for y in range(BOARD_MAX_DIM):
            for x in range(BOARD_MAX_DIM):
                cell = xy_to_cell(x, y)
                if cell == CELL_OOB:
                    result += ' '
                elif self.cells[cell] == NOBODY:
                    result += '.'
                else:
                    result += str(self.cells[cell])
            result += '\n'
        return result[:-1] # We're gonna print() this, so no training newline in the text data itself.

    def __getitem__(self, cell: int) -> int:
        return self.cells[cell]
