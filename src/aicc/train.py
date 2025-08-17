from aicc.neural_network import NeuralNetwork, choose_best_move_plus, ScoreFunction
from aicc.board import Board, get_opposition_player_no
from aicc.game import Game


# Important: this is a subjective game scoring function. A higher score does not necessarily mean
# the player's position is truly better (i.e. the player is more likely to win). It's mainly
# intended to be used as an informational tool for humans evaluating the performance of models and
# is biased toward a few things which intuitively make sense (to me) about good performance in a
# match.
def score_game(game: Game, player_no: int) -> float:
    if player_no not in [1, 2]:
        raise NotImplementedError(f"Player {player_no} is not supported.")

    total = 0.0

    winner = game.get_winner()
    if winner == player_no:
        # You won, reward greatly.
        total += 400.0
    elif winner is not None:
        # You lost, punish greatly.
        total -= 400.0

    home = Board.get_home_cells(player_no)
    goal = Board.get_goal_cells(player_no)

    for row in range(Board.GRID_DIM):
        for col in range(Board.GRID_DIM):
            cell = game.board.state[row, col]

            if cell == player_no:
                # Reward for pieces further away from home zone.
                if player_no == 1:
                    total += row
                elif player_no == 2:
                    total += Board.GRID_DIM - row - 1

                # Reward for pieces in the goal zone.
                if (row, col) in goal:
                    total += 20.0

            else:
                # Punish for opposing pieces close to home zone.
                if player_no == 1:
                    total -= Board.GRID_DIM - row - 1
                elif player_no == 2:
                    total -= row

                # Punish for opposing pieces in the home zone.
                if (row, col) in home:
                    total -= 20.0

    # Punish for taking more turns to complete the game.
    total -= (game.turn_no // game.num_players) / 5.0

    return total


def play_match(model_1: NeuralNetwork, model_2: NeuralNetwork, score_function: ScoreFunction) -> Game:
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

    return game
