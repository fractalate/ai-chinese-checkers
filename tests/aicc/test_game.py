import pytest

from aicc.game import Game
from aicc.board import Board, get_opposition_player_no

@pytest.mark.parametrize("player_no", [1, 2])
def test_win_when_goal_filled_with_player(player_no):
  game = Game(2)

  assert not game.is_winner(player_no)
  for row, col in Board.get_goal_cells(player_no):
      game.board.state[row, col] = player_no
  assert game.is_winner(player_no)

@pytest.mark.parametrize("player_no", [1, 2])
def test_win_when_goal_filled_with_mixed_players(player_no):
  game = Game(2)

  opposition_player_no = get_opposition_player_no(player_no)

  assert not game.is_winner(player_no)
  for i, (row, col) in enumerate(Board.get_goal_cells(player_no)):
      game.board.state[row, col] = player_no if i % 2 == 0 else opposition_player_no
  assert game.is_winner(player_no)
