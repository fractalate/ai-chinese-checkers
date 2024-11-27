from board import Board

class Game:
    def __init__(self):
        self.board = Board()
        self.player_up = 1 # 1 or 2.
        self.turn_no = 1 # Increases by 1 after a player completes a move.
        self.winner = 0 # Nobody yet.
        self.loser = 0 # Nobody yet.

    def is_valid_move(self, from_cell: int, to_cell: int) -> bool:
        if not self.board.is_valid_move(from_cell, to_cell):
            return False
        return self.player_up == self.board[from_cell]

    def move(self, from_cell: int, to_cell: int):
        self.board.move(from_cell, to_cell)

    def is_winner(self, player: int):
        return False # TODO
