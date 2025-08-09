from aicc.game import Game
from aicc.neural_network import NeuralNetwork
from aicc.train import play_match, score_game

if __name__ == "__main__":
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

    play_match(NeuralNetwork(), NeuralNetwork(), score_function)
