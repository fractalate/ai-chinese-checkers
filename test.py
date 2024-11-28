from typing import List

import torch
import torch.nn as nn

from board import BOARD_MAX_DIM, cell_to_xy
from game import Game, get_opposite_player
from neural_network import (
    NeuralNetwork,
    choose_best_action,
    game_state_to_input_vector,
)

game = Game()

class Player:
    def __init__(self, model, name, player_no):
        self.model = model
        self.name = name
        self.moves = 0
        if player_no != 1 and player_no != 2:
            raise NotImplementedError()
        self.player_no = player_no
        self.winner = False
        self.loser = False

    def score(self, game: Game) -> float:
        total = 0.0
        if self.winner:
            total += 1000.0
        if self.loser:
            total -= 1000.0
        for cell, pawn in enumerate(game.board.cells):
            _, y = cell_to_xy(cell)
            if pawn == self.player_no:
                if self.player_no == 1:
                    total += y
                elif self.player_no == 2:
                    total += BOARD_MAX_DIM - y - 1
                else:
                    pass # Not possible because of the check in the constructor.
            opposition = get_opposite_player(self.player_no)
            if cell in game.board.get_player_start_cells(opposition):
                if game.board[cell] == self.player_no:
                    total += 10.0
        total -= self.moves / 10.0
        return total

def play(game: Game, player1: Player, player2: Player):
    while player1.moves + player2.moves < 300:
        if game.player_up == 1:
            player = player1
        elif game.player_up == 2:
            player = player2
        else:
            raise NotImplementedError()

        inputs = game_state_to_input_vector(game, game.player_up) # Maybe use player.player_no everywhere instead? Would I need more checks for if its a valid move, like if it shouldn't actually be the other players turn?
        outputs = player.model(inputs)

        print()
        best_action = choose_best_action(game, game.player_up, outputs)
        if best_action is None:
            print('Best Action: Do Nothing')
            game.do_nothing()
        else:
            from_cell, to_cell = best_action
            print(f'Best Action: ({cell_to_xy(from_cell)}) to ({cell_to_xy(to_cell)})')
            game.move(from_cell, to_cell)

        player.moves += 1

        game.board.print()

        if game.is_winner(1):
            player1.winner = True
            player2.loser = True
            break
        elif game.is_winner(2):
            player2.winner = True
            player1.loser = True
            break

game = Game()

player1 = Player(
    model=NeuralNetwork(),
    name='Player 1',
    player_no=1,
)

player2 = Player(
    model=NeuralNetwork(),
    name='Player 2',
    player_no=2,
)

play(game, player1, player2)

p1 = player1.score(game)
p2 = player2.score(game)

print(f'{player1.name} - Score: {p1} {player1.moves}')
print(f'{player2.name} - Score: {p2} {player2.moves}')
