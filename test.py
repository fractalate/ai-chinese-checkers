from typing import List

import os
import shutil
import sqlite3
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

def train(gen: int, no_models: int):
    print(f'Training generation {gen=} {no_models=}')
    for model_no_1 in range(1, no_models + 1):
        for model_no_2 in range(1, no_models + 1):
            if model_no_1 == model_no_2:
                continue

            with sqlite3.connect('training.db') as connection:

                cursor = connection.cursor()

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS matches (
                        match_no INTEGER PRIMARY KEY AUTOINCREMENT,
                        gen INTEGER
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS match_players (
                        match_no INTEGER,
                        gen INTEGER,
                        model_no INTEGER,
                        player_no INTEGER,
                        winner BOOLEAN,
                        loser BOOLEAN,
                        score DOUBLE
                    )
                ''')

                connection.commit()

                cursor.execute('''
                    SELECT match_no FROM matches JOIN match_players a USING (match_no) JOIN match_players b USING (match_no)
                        WHERE matches.gen = ? AND a.model_no = ? AND b.model_no = ?
                ''', (gen, model_no_1, model_no_2))

                if len(cursor.fetchall()) > 0:
                    continue

                game = Game()

                try:
                    os.mkdir('models')
                except FileExistsError:
                    pass
                try:
                    os.mkdir(os.path.join('models', str(gen)))
                except FileExistsError:
                    pass

                model_1_path = os.path.join('models', str(gen), str(model_no_1) + '.pth')
                if os.path.isfile(model_1_path):
                    model_1 = NeuralNetwork() # TODO How do I avoid making these initial weights?
                    model_1.load_state_dict(torch.load(model_1_path, weights_only=True))
                elif gen == 1:
                    model_1 = NeuralNetwork()
                    torch.save(model_1.state_dict(), model_1_path)
                else:
                    raise Exception(f'Expected model {gen=} model_no={model_no_1} to be previously created.')

                model_2_path = os.path.join('models', str(gen), str(model_no_2) + '.pth')
                if os.path.isfile(model_2_path):
                    model_2 = NeuralNetwork() # TODO How do I avoid making these initial weights?
                    model_2.load_state_dict(torch.load(model_2_path, weights_only=True))
                elif gen == 1:
                    model_2 = NeuralNetwork()
                    torch.save(model_2.state_dict(), model_2_path)
                else:
                    raise Exception(f'Expected model {gen=} model_no={model_no_2} to be previously created.')

                player1 = Player(
                    model=model_1,
                    name='Player 1',
                    player_no=1,
                )

                player2 = Player(
                    model=model_2,
                    name='Player 2',
                    player_no=2,
                )

                play(game, player1, player2)

                p1 = player1.score(game)
                p2 = player2.score(game)

                print(f'{player1.name} - Score: {p1} {player1.moves}')
                print(f'{player2.name} - Score: {p2} {player2.moves}')

                cursor.execute('''
                    INSERT INTO matches(gen)
                        VALUES(?)
                        RETURNING match_no
                ''', (gen ,))
                rows = cursor.fetchall()

                for (match_no, ) in rows:
                    cursor.execute('''
                        INSERT INTO match_players(match_no, gen, model_no, player_no, winner, loser, score)
                            VALUES(?, ?, ?, ?, ?, ?, ?)
                    ''', (match_no, gen, model_no_1, player1.player_no, player1.winner, player1.loser, player1.score(game)))

                    cursor.execute('''
                        INSERT INTO match_players(match_no, gen, model_no, player_no, winner, loser, score)
                            VALUES(?, ?, ?, ?, ?, ?, ?)
                    ''', (match_no, gen, model_no_2, player2.player_no, player2.winner, player2.loser, player2.score(game)))

                connection.commit()

                try:
                    os.mkdir('matches')
                except FileExistsError:
                    pass

                with open(os.path.join('matches', str(match_no) + '.txt'), 'w') as fout:
                    fout.write(game.board.dumps() + '\n')

def create_generation(gen, no_models):
    print(f'Creating generation {gen=}')

    if no_models != 10:
        raise NotImplementedError()

    if gen == 1:
        return # should be created by train()
    
    try:
        os.mkdir(os.path.join('models', str(gen)))
    except FileExistsError:
        pass

    with sqlite3.connect('training.db') as connection:
        cursor = connection.cursor()

        top_model_rows = cursor.execute('''
            SELECT model_no, SUM(score) FROM match_players WHERE gen = ? GROUP BY gen, model_no ORDER BY 2, model_no DESC
        ''', (gen - 1,)).fetchall()
        assert len(top_model_rows) > 0
        top_model_nos: List[int] = []
        for (model_no, _) in top_model_rows[:3]:
            top_model_nos.append(model_no)
        del model_no

        new_model_no = 1

        # The top model survives.
        print(f'Top model survives: {(gen - 1, top_model_nos[0])} -> {(gen, new_model_no)}')

        new_model_path = os.path.join('models', str(gen), str(new_model_no) + '.pth')
        if os.path.isfile(new_model_path):
            print('Skipping.')
        else:
            top_model_path = os.path.join('models', str(gen - 1), str(top_model_nos[0]) + '.pth')
            shutil.copy(top_model_path, new_model_path)

        new_model_no += 1

        # Breeding.
        for i, model_no_1 in enumerate(top_model_nos):
            for j, model_no_2 in enumerate(top_model_nos):
                if j < i:
                    continue
                if model_no_1 == model_no_2:
                    continue

                print(f'Breeding: {(gen - 1, model_no_1)} + {(gen - 1, model_no_2)} -> {(gen, new_model_no)}.')

                new_model_path = os.path.join('models', str(gen), str(new_model_no) + '.pth')
                if os.path.isfile(new_model_path):
                    print('Skipping.')
                else:
                    model_1_path = os.path.join('models', str(gen - 1), str(model_no_1) + '.pth')
                    model_1 = NeuralNetwork() # TODO How do I avoid making these initial weights?
                    model_1.load_state_dict(torch.load(model_1_path, weights_only=True))

                    model_2_path = os.path.join('models', str(gen - 1), str(model_no_2) + '.pth')
                    model_2 = NeuralNetwork() # TODO How do I avoid making these initial weights?
                    model_2.load_state_dict(torch.load(model_2_path, weights_only=True))

                    new_model = NeuralNetwork()
                    with torch.no_grad():
                        for param1, param2, param_combined in zip(model_1.parameters(), model_2.parameters(), new_model.parameters()):
                            param_combined.data = (param1.data + param2.data) / 2.0

                    torch.save(new_model.state_dict(), new_model_path)

                new_model_no += 1

        # Variation.
        for i in range(no_models - new_model_no + 1):
            base_model_no = top_model_nos[i % len(top_model_nos)]

            print(f'Varying: {(gen - 1, base_model_no)} -> {(gen, new_model_no)}.')

            new_model_path = os.path.join('models', str(gen), str(new_model_no) + '.pth')
            if os.path.isfile(new_model_path):
                print('Skipping.')
            else:
                base_model_path = os.path.join('models', str(gen - 1), str(base_model_no) + '.pth')
                base_model = NeuralNetwork() # TODO How do I avoid making these initial weights?
                base_model.load_state_dict(torch.load(base_model_path, weights_only=True))

                with torch.no_grad():
                    for param in base_model.parameters():
                        param.add_(torch.randn_like(param) * 0.01)

                torch.save(base_model.state_dict(), new_model_path)

            new_model_no += 1

for gen in range(1, 100):
    train(gen, 10)
    create_generation(gen + 1, 10)
