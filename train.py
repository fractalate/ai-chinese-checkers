from typing import List, Tuple

from datetime import datetime
import os
import sqlite3
from uuid import uuid4

import torch
import torch.nn as nn

from neural_network import NeuralNetwork

class Trainer:
    def __init__(self, name: str):
        self.trainer_version: str = 'basic_trainer'
        self.name: str = name

        self.models_per_generation: int = 100

        base_dir = self.get_base_dir()
        self.connection: sqlite3.Connection = sqlite3.connect(os.path.join(base_dir, 'trainer.db'), timeout=30)

        self.setup()

    def get_base_dir(self) -> str:
        # 'out/basic_trainer/trial_103'
        base_dir = os.path.join('out', self.trainer_version, self.name)
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    # ---------------------------------------------------------------------------------------------
    # SETUP
    # ---------------------------------------------------------------------------------------------

    def setup(self):
        # cursor comes from inside the transaction which follows.
        def set_trainer_detail(key: str, value: str, overwrite: bool = False, must_match_stored: bool = False) -> str:  # Getter also.
            for (stored_value, ) in cursor.execute('''SELECT value FROM trainer_details WHERE key = ?''', (key, )).fetchall():
                if must_match_stored and stored_value != value:
                    raise Exception(f'Trainer({self.trainer_version=}, {self.name=}) cannot load when {key}={value}.')
            if overwrite:
                (result_value, ) = cursor.execute('''INSERT INTO trainer_details(key, value) VALUES(?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value RETURNING value''', (key, value)).fetchone()
            else:
                # Effectively a DO NOTHING with the SET value = value (contrast with SET value = excluded.value).
                # Use this pattern so RETURNING behaves.
                (result_value, ) = cursor.execute('''INSERT INTO trainer_details(key, value) VALUES(?, ?) ON CONFLICT (key) DO UPDATE SET value = value RETURNING value''', (key, value)).fetchone()
            return result_value

        with self.connection:
            self.connection('''BEGIN EXCLUSIVE''')
            cursor = self.connection.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trainer_details (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')

            set_trainer_detail('trainer_version', self.trainer_version, must_match_stored=True)
            set_trainer_detail('creation_time', str(datetime.now().astimezone()))
            set_trainer_detail('last_open_time', str(datetime.now().astimezone()), overwrite=True)

            self.models_per_generation = set_trainer_detail('models_per_generation', self.models_per_generation)

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    model_no INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    gen INTEGER
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_provenances (
                    model_no INTEGER PRIMARY KEY,
                    description STRING
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_pedigrees (
                    model_no INTEGER,
                    parent_model_no INTEGER,
                    role STRING
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS generations (
                    gen INTEGER PRIMARY KEY AUTOINCREMENT,
                    population INTEGER,
                    complete BOOLEAN
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS matches (
                    match_no INTEGER PRIMARY KEY,
                    gen INTEGER PRIMARY,
                    complete BOOLEAN
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS promised_matches (
                    gen INTEGER,
                    model_no_1 INTEGER,
                    model_no_2 INTEGER
                )
            ''')

    # ---------------------------------------------------------------------------------------------
    # GENERATION MANAGEMENT
    # ---------------------------------------------------------------------------------------------

    def _load_gen(self, cursor: sqlite3.Cursor) -> Tuple[bool, None | Tuple[int, int]]:
        rows = cursor.execute('''SELECT gen, population FROM generations WHERE NOT complete ORDER BY gen DESC''').fetchall()
        assert len(rows) <= 1  # If this fails, something is up with generation creation or finalizing.
        for (gen, population) in rows:
            print(f'_load_gen() found {gen=}, {population=}')
            return True, (gen, population)
        return False, None

    def _create_next_gen(self, cursor: sqlite3.Cursor) -> Tuple[int, int]:
        gen, population = cursor.execute('''
            INSERT INTO generations(population, complete) VALUES(?, ?)
                RETURNING gen, population
        ''', (self.models_per_generation, False)).fetchone()
        print(f'_create_next_gen() created {gen=}, {population=}')
        return gen, population

    def load_gen(self) -> Tuple[int, int]:
        with self.connection:
            self.connection.execute('''BEGIN EXCLUSIVE''')
            cursor = self.connection.cursor()

            loaded, gen_info = self._load_gen(cursor)
            if not loaded:
                gen_info = self._create_next_gen(cursor)

            return gen_info

    # ---------------------------------------------------------------------------------------------
    # BREEDING
    # ---------------------------------------------------------------------------------------------

    def ensure_ample_population(self, gen: int, population: int):
        while True:
            with self.connection:
                cursor = self.connection.cursor()

                (model_count, ) = cursor.execute('''
                    SELECT COUNT(*) AS model_count FROM models WHERE gen = ?
                ''', (gen, )).fetchone()
                if model_count >= population:
                    break

            if gen == 1:
                filename, provenance, pedigrees = self._create_new_model(gen)
            else:
                filename, provenance, pedigrees = self._breed_new_model(gen)

            with self.connection:
                self.connection('''BEGIN EXCLUSIVE''')
                cursor = self.connection.cursor()

                (model_count, ) = cursor.execute('''
                    SELECT COUNT(*) AS model_count FROM models WHERE gen = ?
                ''', (gen, )).fetchone()

                if model_count >= population:
                    print('ensure_ample_population() discarded a model due to race conditions')
                    self._unbreed_model(gen, filename)
                    break

                (model_no, ) = cursor.execute('''
                    INSERT INTO models(filename, gen) VALUES(?, ?) RETURNING model_no
                ''', (filename, gen, )).fetchone()

                cursor.execute('''
                    INSERT INTO model_provenances(model_no, description) VALUES(?, ?)
                ''', (model_no, provenance))

                for (parent_model_no, role) in pedigrees:
                    cursor.execute('''
                        INSERT INTO model_pedigrees(model_no, parent_model_no, role) VALUES(?, ?, ?)
                    ''', (model_no, parent_model_no, role))

                print(f'ensure_ample_population() inserted model {filename=} {gen=} {model_no=}')

    def _create_new_model(self, gen: int):
        base_dir = self.get_base_dir()
        os.makedirs(os.path.join(base_dir, 'models', str(gen)), exist_ok=True)
        filename = str(uuid4()) + '.pth'
        provenance = 'new untrained model'
        pedigrees = []  # No parentage.
        model = NeuralNetwork()
        torch.save(model.state_dict(), os.path.join(base_dir, 'models', str(gen), filename))
        print(f'_create_new_model() created a new model {filename=} {provenance=} {pedigrees=}')
        return filename, provenance, pedigrees

    def _breed_new_model(self, gen: int):
        raise NotImplementedError()

    def _unbreed_model(self, gen: int, filename: str):
        base_dir = self.get_base_dir()
        os.unlink(os.path.join(base_dir, 'models', str(gen), filename))

    # ---------------------------------------------------------------------------------------------
    # MATCHES
    # ---------------------------------------------------------------------------------------------

    def do_matches(self, gen: int):
        raise NotImplementedError()

    # ---------------------------------------------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------------------------------------------

    def train(self):
        gen, population = self.load_gen()
        self.ensure_ample_population(gen, population)
        self.do_matches(gen)
        print('Nothing to do.')

def main():
    trainer = Trainer('sample')
    trainer.train()

if __name__ == '__main__':
    main()

"""

def _make_dirs_for_model(gen: int):
    pass

def load_model(gen: int, model_no: int) -> NeuralNetwork:
    pass

def save_model(model: NeuralNetwork):
    pass

def get_models_for_generation(gen: int):
    model_nos = []
    gen_path = os.path.join('generations', str(gen))
    for model in os.listdir(gen_path):
        if model.endswith('.pth'):
            model_no = int(model[:-4])
            model_nos.append(model_no)
    return model_nos

def train_generation(gen: int):
    model_nos = get_models_for_generation(gen)
    if gen == 1:
        new_model_no = 1
        while len(model_nos) < 10:
            while new_model_no in model_nos:
                new_model_no += 1
            model_nos.append(new_model_no)
            print(f'Creating model {new_model_no} for generation {gen}...')
            model = NeuralNetwork()
            where = os.path.join('generations', str(gen), str(new_model_no) + '.pth')
            torch.save(model.state_dict(), where)
            print(f'Model saved to {where}')

def play_a_match():
    player1 = Player(
        model=NeuralNetwork(),
        name='Player 1',
    )

    player2 = Player(
        model=NeuralNetwork(),
        name='Player 2',
    )

    state = board.new_board_state()
    winner = None

    for i in range(30):
        if is_game_won_by_friendly(state):
            winner = player1
            player1.winner = True
            player2.loser = True
            break

        xplayer = player1 if i % 2 == 0 else player2
        oplayer = player2 if i % 2 == 0 else player1

        inputs = board_state_to_input_vector(state)
        outputs = player1.model(inputs)

        print()
        print(f'X - {xplayer.name}')
        print(f'O - {oplayer.name}')

        best_action = choose_best_action(state, outputs)
        if best_action is None:
            print('Best Action: Do Nothing')
        else:
            fc, tc = best_action
            fx, fy = board.cell_to_xy(fc)
            tx, ty = board.cell_to_xy(tc)
            print(f'Best Action: ({fx}, {fy}) to ({tx}, {ty})')

            state[tc] = state[fc]
            state[fc] = board.EMPTY
            player1.moves += 1

        if is_game_won_by_friendly(state):
            winner = player1
            player1.winner = True
            player2.loser = True
            break

        board.print_board(state if i % 2 == 0 else board.flip_board_state(state))
        state = board.flip_board_state(state)
        player1, player2 = player2, player1

    if winner is None:
        print('The only way to win is to not play the game. These models did not "play". (Draw)')
    else:
        print(f'Winner: {winner}!')

    p1 = player1.score(state)
    p2 = player2.score(board.flip_board_state(state))

    print(f'{player1.name} - Score: {p1}')
    print(f'{player2.name} - Score: {p2}')
"""
