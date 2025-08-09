from typing import Optional
from aicc.board import Board

class Game:
    def __init__(self, num_players: int):
        self.board: Board = Board(skip_init=True)
        self.num_players: int = num_players
        self.player_up: int = 1
        self.turn_no: int = 1  # Increases by 1 after a player completes a move.
        self.winner: int = 0  # Nobody yet.
        self.loser: int = 0  # Nobody yet.
        self.board.setup_board(num_players)

    def is_valid_move(
        self, from_row: int, from_col: int, to_row: int, to_col: int
    ) -> bool:
        if self.player_up != self.board.state[from_row, from_col]:
            return False
        return self.board.is_valid_move(from_row, from_col, to_row, to_col)

    def move(self, from_row: int, from_col: int, to_row: int, to_col: int):
        self.board.move(from_row, from_col, to_row, to_col)
        self.advance_turn()

    def advance_turn(self):
        self.player_up = self.player_up % self.num_players + 1
        self.turn_no += 1

    def get_winner(self) -> Optional[int]:
        for player_no in range(1, self.num_players + 1):
            if self.is_winner(player_no):
                return player_no
        return None

    def is_winner(self, player_no: int):
        has_player_pawn = False
        count_cells_filled = 0
        goal_cells = Board.get_goal_cells(player_no)
        for row, col in goal_cells:
            if self.board.state[row, col] != Board.NOBODY:
                count_cells_filled += 1
            if self.board.state[row, col] == player_no:
                has_player_pawn = True
        return has_player_pawn and count_cells_filled == len(goal_cells)
