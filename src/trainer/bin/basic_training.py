import os
from typing import List

import torch

from aicc.board import get_opposition_player_no
from aicc.breed import (
    breed_mutate,
    breed_stochastic_crossover_mutation,
    breed_weighted_average,
)
from aicc.game import Game
from aicc.neural_network import NeuralNetwork
from aicc.train import play_match, score_game, ScoreFunction


NUMBER_OF_MODELS = 10
NUMBER_OF_SURVIVORS = 3


_model_descriptor_id_sequence = 0
def next_model_descriptor_id():
    global _model_descriptor_id_sequence
    _model_descriptor_id_sequence += 1
    return _model_descriptor_id_sequence


class ModelDescriptor:
    def __init__(self, model: NeuralNetwork):
        self.id: int = next_model_descriptor_id()
        self.model: NeuralNetwork = model


class ModelScoreSheet:
    def __init__(self, model_descriptor: ModelDescriptor):
        self.model_descriptor: ModelDescriptor = model_descriptor
        self.total_score: float = 0.0


def play_matches_and_rank_models(model_descriptors: List[ModelDescriptor], score_function: ScoreFunction):
    model_score_sheets = [ModelScoreSheet(model_descriptor) for model_descriptor in model_descriptors]
    for model_descriptor in model_score_sheets:
        model_descriptor.total_score = 0.0
    for model_score_sheet_1 in model_score_sheets:
        for model_score_sheet_2 in model_score_sheets:
            if model_score_sheet_2 is model_score_sheet_1: continue
            game = play_match(model_score_sheet_1.model_descriptor.model, model_score_sheet_2.model_descriptor.model, score_function)
            model_score_sheet_1.total_score += score_game(game, 1)
            model_score_sheet_2.total_score += score_game(game, 2)
    return [
        model_score_sheet.model_descriptor for model_score_sheet in
            sorted(model_score_sheets, key=lambda model_score_sheet: model_score_sheet.total_score, reverse=True)
    ]


def perform_training_round(model_descriptors: List[ModelDescriptor]):
    ranked_model_descriptors = play_matches_and_rank_models(model_descriptors, score_function)

    for position, model_descriptor in enumerate(ranked_model_descriptors, start=1):
        print(f"{position}: Model {model_descriptor.id}")

    return ranked_model_descriptors


def cull_and_breed(ranked_model_descriptors: List[ModelDescriptor]) -> List[ModelDescriptor]:
    survivors = ranked_model_descriptors[:NUMBER_OF_SURVIVORS]

    population = [] + survivors

    for i, parent_1 in enumerate(survivors[:-1]):
        for parent_2 in survivors[i+1:]:
            population.append(ModelDescriptor(breed_weighted_average(parent_1.model, parent_2.model)))
            population.append(ModelDescriptor(breed_stochastic_crossover_mutation(parent_1.model, parent_2.model)))

    i = 0
    while len(population) < NUMBER_OF_MODELS:
        parent = survivors[i % len(survivors)]
        population.append(ModelDescriptor(breed_mutate(parent)))
        i += 1

    if len(population) > NUMBER_OF_MODELS:
        print(f"TOO MANY MODELS IN POPULATION! Found {len(population)} but can't exceed {NUMBER_OF_MODELS}. Dropping excess.")
        population = population[:NUMBER_OF_MODELS]

    return population


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

    model_descriptors = [ModelDescriptor(NeuralNetwork()) for _ in range(NUMBER_OF_MODELS)]
    while True:
        ranked_model_descriptors = perform_training_round(model_descriptors)

        # This is dumb since it overwrites the old ones... but I don't want to take up a bunch of
        # space with this training data just yet.
        best_model = ranked_model_descriptors[0].model
        os.makedirs("out", exist_ok=True)
        torch.save(best_model.state_dict(), os.path.join("out", "basic_training_best.model"))

        model_descriptors = cull_and_breed(ranked_model_descriptors)
