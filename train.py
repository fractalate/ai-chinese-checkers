from typing import List, Tuple

from datetime import datetime
import os
import sqlite3
import time
from uuid import uuid4

import torch
import torch.nn as nn

from game import Game
from play import Player, play
from neural_network import NeuralNetwork

class Trainer:
    def __init__(self, name: str):
        self.trainer_version: str = 'basic_trainer'
        self.name: str = name

        self.models_per_generation: int = 10  # TODO bump up to 23 so there are like 500 matches in a generation
        self.top_models_to_keep_per_generation: int = 2  # TODO bump up to 23 so there are like 500 matches in a generation

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
                    gen INTEGER,
                    complete BOOLEAN
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS match_models (
                    match_no INTEGER,
                    player_no INTEGER,
                    model_no INTEGER
                )
            ''')

            cursor.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS idx_match_models_players ON match_models(match_no, player_no)
            ''')

            # Filled when the match is complete.
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS match_scores (
                    match_no INTEGER,
                    player_no INTEGER,
                    model_no INTEGER,
                    winner BOOLEAN,
                    loser BOOLEAN,
                    score FLOAT
                )
            ''')

            cursor.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS idx_match_scores_players ON match_scores(match_no, player_no)
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

    def complete_generation(self, gen: int):
        with self.connection:
            self.connection.execute('''BEGIN EXCLUSIVE''')
            cursor = self.connection.cursor()

            # Multiple processes will come through here and do this. But that's fine.
            cursor.execute('''UPDATE generations SET complete = TRUE WHERE gen = ?''', (gen, ))

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
        while True:
            found, match_info = self.find_needed_match(gen)
            if not found:
                break
            match_no, model_no_1, model_no_2 = match_info
            game, player1, player2 = self.do_match(match_no, model_no_1, model_no_2)
            self.complete_match(gen, game, match_no, model_no_1, player1, model_no_2, player2)

    def find_needed_match(self, gen: int) -> Tuple[bool, None | Tuple[int, int, int]]:
        with self.connection:
            self.connection.execute('''BEGIN EXCLUSIVE''')
            cursor = self.connection.cursor()

            # First, try to get a match that hasn't been previously promised.
            rows = cursor.execute('''
                SELECT * FROM (
                    SELECT a.model_no, b.model_no FROM models a, models b
                        WHERE a.gen = ? AND b.gen = ? AND a.model_no != b.model_no
                    EXCEPT
                    SELECT mm1.model_no, mm2.model_no
                        FROM matches, match_models mm1, match_models mm2
                        WHERE matches.match_no = mm1.match_no AND mm1.player_no = 1
                        AND matches.match_no = mm2.match_no AND mm2.player_no = 2
                ) t
                    ORDER BY RANDOM()
                    LIMIT 1
            ''', (gen, gen, )).fetchall()

            if len(rows) > 0:
                for (model_no_1, model_no_2) in rows:
                    break

                (match_no, ) = cursor.execute('''
                    INSERT INTO matches(gen, complete) VALUES(?, ?) RETURNING match_no
                ''', (gen, False)).fetchone()

                cursor.execute('''
                    INSERT INTO match_models(match_no, model_no, player_no) VALUES(?, ?, ?)
                ''', (match_no, model_no_1, 1))

                cursor.execute('''
                    INSERT INTO match_models(match_no, model_no, player_no) VALUES(?, ?, ?)
                ''', (match_no, model_no_2, 2))

                print(f'find_needed_match() created new match {match_no=}, {model_no_1=}, {model_no_2=}')

                return True, (match_no, model_no_1, model_no_2)

            # Then, if there were no such, get one that has been promised, but is not complete.
            rows = cursor.execute('''
                SELECT matches.match_no, mm1.model_no, mm2.model_no
                    FROM matches, match_models mm1, match_models mm2
                    WHERE gen = ?
                    AND matches.match_no = mm1.match_no AND mm1.player_no = 1
                    AND matches.match_no = mm2.match_no AND mm2.player_no = 2
                    AND NOT complete
                ORDER BY RANDOM()
                LIMIT 1
            ''', (gen, )).fetchall()

            for (match_no, model_no_1, model_no_2) in rows:
                print(f'find_needed_match() gave old incomplete match {match_no=}, {model_no_1=}, {model_no_2=}')
                return True, (match_no, model_no_1, model_no_2)
            return False, None

    def _load_model(self, model_no):
        with self.connection:
            cursor = self.connection.cursor()

            (gen, filename, ) = cursor.execute('''
                SELECT gen, filename FROM models WHERE model_no = ?
            ''', (model_no ,)).fetchone()

            base_dir = self.get_base_dir()
            
            model = NeuralNetwork()  # TODO How do I prevent initializing this before replacing its contents?
            model_path = os.path.join(base_dir, 'models', str(gen), filename)
            model.load_state_dict(torch.load(model_path, weights_only=True))

            return model

    def do_match(self, match_no: int, model_no_1: int, model_no_2: int):
        model_1 = self._load_model(model_no_1)
        player1 = Player(model_1, 'Player 1', 1)
        model_2 = self._load_model(model_no_2)
        player2 = Player(model_2, 'Player 2', 2)
        game = Game()
        print(f'do_match() is playing {match_no=} {model_no_1=} {model_no_2=}')
        play(game, player1, player2)
        return game, player1, player2

    def complete_match(self, gen: int, game: Game, match_no: int, model_no_1: int, player1: Player, model_no_2: int, player2: Player):
        with self.connection:
            self.connection.execute('''BEGIN EXCLUSIVE''')
            cursor = self.connection.cursor()

            (complete, ) = cursor.execute('''
                SELECT complete FROM matches WHERE match_no = ?
            ''', (match_no, )).fetchone()

            if complete:
                print(f'complete_match() discarded match {match_no=} {model_no_1=} {model_no_2=}')
                return

            base_dir = self.get_base_dir()
            os.makedirs(os.path.join(base_dir, 'matches', str(gen)), exist_ok=True)
            with open(os.path.join(base_dir, 'matches', str(gen), str(match_no) + '.txt'), 'w') as fout:
                fout.write(game.board.dumps() + '\n')

            player1_score = player1.score(game)
            player2_score = player2.score(game)

            cursor.execute('''
                UPDATE matches SET complete = TRUE
                    WHERE match_no = ?
            ''', (match_no, ))

            cursor.execute('''
                INSERT INTO match_scores(match_no, player_no, model_no, winner, loser, score)
                    VALUES(?, ?, ?, ?, ?, ?)
            ''', (match_no, 1, model_no_1, player1.winner, player1.winner, player1_score))

            cursor.execute('''
                INSERT INTO match_scores(match_no, player_no, model_no, winner, loser, score)
                    VALUES(?, ?, ?, ?, ?, ?)
            ''', (match_no, 2, model_no_2, player2.winner, player2.winner, player2_score))

            print(f'complete_match() recorded match {match_no=} {model_no_1=} {player1_score=} (W/L {player1.winner} {player1.loser}) {model_no_2=} {player2_score=} (W/L {player2.winner} {player2.loser})')

    # ---------------------------------------------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------------------------------------------

    def train(self):
        while True:
            gen, population = self.load_gen()
            self.ensure_ample_population(gen, population)
            self.do_matches(gen)
            self.complete_generation(gen)
            time.sleep(1)

def main():
    trainer = Trainer('sample')
    trainer.train()

if __name__ == '__main__':
    main()
