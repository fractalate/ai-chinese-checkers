from aicc.neural_network import NeuralNetwork, choose_best_move_plus
from aicc.board import Board
from aicc.game import Game


def score_game(game: Game, player_no: int) -> float:
    total = 0.0
    winner = game.get_winner()
    if winner == player_no:
        # You won.
        total += 200.0
    elif winner is not None:
        # You lost.
        total -= 200.0

    goal = Board.get_goal_cells(player_no)

    for row in range(Board.GRID_DIM):
        for col in range(Board.GRID_DIM):
            cell = game.board.state[row, col]
            if cell == player_no:
                if player_no == 1:
                    total += row
                elif player_no == 2:
                    total += Board.GRID_DIM - row - 1

                if (row, col) in goal:
                    total += 10.0

    total -= (game.turn_no // game.num_players) / 5.0

    return total


def play_match(model_1: NeuralNetwork, model_2: NeuralNetwork) -> Game:
    def score_function(
        game: Game,
        from_row: int,
        from_col: int,
        from_cell_score: float,
        to_row: int,
        to_col: int,
        to_cell_score: float,
    ):
        return score_game(game, game.player_up) + from_cell_score + to_cell_score

    game = Game(num_players=2)

    player_no = 1
    models = [None, model_1, model_2]

    while not game.is_winner(game.player_up) and game.turn_no < 150:
        move = choose_best_move_plus(models[player_no], game, score_function)
        if not move:
            game.advance_turn()
        else:
            game.move(*move[0], *move[1])
        player_no = game.player_up

        print(game.board.dumps())
        print('player 1:', score_game(game, 1))
        print('player 2:', score_game(game, 2))
        print()


if __name__ == "__main__":
    play_match(NeuralNetwork(), NeuralNetwork())
