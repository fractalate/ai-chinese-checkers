from board import Board, NOBODY

class Game:
    def __init__(self):
        self.board = Board()
        self.player_up = 1 # 1 or 2.
        self.turn_no = 1 # Increases by 1 after a player completes a move.
        self.winner = 0 # Nobody yet.
        self.loser = 0 # Nobody yet.

    def is_valid_move(self, from_cell: int, to_cell: int) -> bool:
        if not self.board.is_valid_move(from_cell, to_cell, backtracking_cells=None):
            return False
        return self.player_up == self.board[from_cell]
    
    def is_valid_no_action(self) -> bool:
        return False # TODO Only relevant when multi-hopping.

    def move(self, from_cell: int, to_cell: int):
        self.board.move(from_cell, to_cell)
        self.player_up = 3 - self.player_up # TODO Somebody stop me.

    def do_nothing(self):
        self.player_up = 3 - self.player_up # TODO Somebody stop me.

    def is_winner(self, player: int):
        opposition = get_opposite_player(player)
        has_player_pawn = False
        count_cells_filled = 0
        goal_cells = self.board.get_player_start_cells(opposition)
        for cell in goal_cells:
            if self.board[cell] == player:
                has_player_pawn = True
            if self.board[cell] != NOBODY:
                count_cells_filled += 1
        return has_player_pawn and count_cells_filled == len(goal_cells)

def get_opposite_player(player: int) -> int:
    if player == 1:
        return 2
    if player == 2:
        return 1
    raise NotImplementedError()
