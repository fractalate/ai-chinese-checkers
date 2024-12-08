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

        self.models_per_generation: int = 12  # TODO bump up to 23 so there are like 500 matches in a generation
        self.top_models_to_keep_per_generation: int = 3  # TODO bump up too
        self.lesser_models_to_breed_with_top_models: int = 2

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

            self.models_per_generation = int(set_trainer_detail('models_per_generation', self.models_per_generation))
            self.top_models_to_keep_per_generation = int(set_trainer_detail('top_models_to_keep_per_generation', self.top_models_to_keep_per_generation))
            self.lesser_models_to_breed_with_top_models = int(set_trainer_detail('lesser_models_to_breed_with_top_models', self.lesser_models_to_breed_with_top_models))

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

            # gen is the target generation.
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_plans (
                    model_plan_no INTEGER PRIMARY KEY AUTOINCREMENT,
                    gen INTEGER,
                    action TEXT,
                    model_no_1 INTEGER,
                    model_no_2 INTEGER,
                    promised BOOLEAN,
                    complete BOOLEAN
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

    def complete_generation(self, gen: int):
        with self.connection:
            self.connection.execute('''BEGIN EXCLUSIVE''')
            cursor = self.connection.cursor()

            (complete, ) = cursor.execute('''SELECT complete FROM generations WHERE gen = ?''', (gen, )).fetchone()
            if complete:
                print('complete_generation() too late to set generation complete')
            else:
                # Multiple processes will come through here and do this. But that's fine.
                cursor.execute('''UPDATE generations SET complete = TRUE WHERE gen = ?''', (gen, ))

                models = cursor.execute('''
                    select m.model_no, sum(ms.score) as score
                        from models m join match_scores ms on ms.model_no = m.model_no
                        where m.gen = ? group by m.model_no order by 2 DESC
                ''', (gen, )).fetchall()
                models = list(models)

                top_models = models[:self.top_models_to_keep_per_generation]
                top_model_nos = []
                for (model_no, _) in top_models:
                    top_model_nos.append(model_no)
                lesser_models = models[self.top_models_to_keep_per_generation:]
                lesser_model_nos = []
                for (model_no, _) in lesser_models:
                    lesser_model_nos.append(model_no)

                count_models_planned = 0

                for model_no_1 in top_model_nos:
                    cursor.execute('''
                        INSERT INTO model_plans(gen, action, model_no_1, promised, complete) values(?, ?, ?, ?, ?)
                    ''', (gen + 1, 'copy', model_no_1, False, False))
                    count_models_planned += 1
                    print(f'complete_generation() planned copy of top model {model_no_1=}')

                for model_no_1 in top_model_nos:
                    for model_no_2 in top_model_nos:
                        if model_no_1 == model_no_2:
                            continue
                        cursor.execute('''
                            INSERT INTO model_plans(gen, action, model_no_1, model_no_2, promised, complete) values(?, ?, ?, ?, ?, ?)
                        ''', (gen + 1, 'breed_crossover_parameters', model_no_1, model_no_2, False, False))
                        count_models_planned += 1
                        print(f'complete_generation() planned breed crossover parameters of top models {model_no_1=} {model_no_2=}')

                for model_no_1 in top_model_nos:
                    for model_no_2 in lesser_model_nos[:self.lesser_models_to_breed_with_top_models]:
                        cursor.execute('''
                            INSERT INTO model_plans(gen, action, model_no_1, model_no_2, promised, complete) values(?, ?, ?, ?, ?, ?)
                        ''', (gen + 1, 'breed_crossover_parameters', model_no_1, model_no_2, False, False))
                        count_models_planned += 1
                        print(f'complete_generation() planned breed crossover parameters of top model with lesser {model_no_1=} {model_no_2=}')

                while count_models_planned < self.models_per_generation:
                    for (model_no_1, _) in models:
                        cursor.execute('''
                            INSERT INTO model_plans(gen, action, model_no_1, promised, complete) values(?, ?, ?, ?, ?)
                        ''', (gen + 1, 'mutate', model_no_1, False, False))
                        count_models_planned += 1
                        print(f'complete_generation() planned mutate of model {model_no_1=}')

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
                model_info = self._create_new_model(gen)
            else:
                model_info = self._breed_new_model(gen)

            if model_info:
                filename, provenance, pedigrees = model_info

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

    def _breed_new_model_copy(self, model_no: int, to_gen: int):
        base_dir = self.get_base_dir()
        os.makedirs(os.path.join(base_dir, 'models', str(to_gen)), exist_ok=True)
        filename = str(uuid4()) + '.pth'
        provenance = 'copy'
        pedigrees = [(model_no, 'source')]
        model = self._load_model(model_no)
        torch.save(model.state_dict(), os.path.join(base_dir, 'models', str(to_gen), filename))
        print(f'_breed_new_model_copy() copied {model_no=} to {filename=} {provenance=} {pedigrees=}')
        return filename, provenance, pedigrees

    def _breed_new_model_mutate(self, model_no: int, to_gen: int):
        base_dir = self.get_base_dir()
        os.makedirs(os.path.join(base_dir, 'models', str(to_gen)), exist_ok=True)
        filename = str(uuid4()) + '.pth'
        provenance = 'mutate'
        pedigrees = [(model_no, 'ancestor')]
        model = self._load_model(model_no)
        for param in model.parameters():
            if torch.rand(1).item() < 0.05:
                noise = torch.randn_like(param.data) * 0.01
                param.data += noise
        torch.save(model.state_dict(), os.path.join(base_dir, 'models', str(to_gen), filename))
        print(f'_breed_new_model_mutate() mutated {model_no=} to {filename=} {provenance=} {pedigrees=}')
        return filename, provenance, pedigrees

    def _breed_new_model_breed_crossover_parameters(self, model_no_1: int, model_no_2: int, to_gen: int):
        base_dir = self.get_base_dir()
        os.makedirs(os.path.join(base_dir, 'models', str(to_gen)), exist_ok=True)
        filename = str(uuid4()) + '.pth'
        provenance = 'breed_crossover_parameters'
        pedigrees = [(model_no_1, 'parent'), (model_no_2, 'parent')]
        model_1 = self._load_model(model_no_1)
        model_2 = self._load_model(model_no_2)
        for param_1, param_2 in zip(model_1.parameters(), model_2.parameters()):
            mask = torch.rand_like(param_1.data) < 0.5
            param_1.data[mask] = param_2.data[mask]
        torch.save(model_1.state_dict(), os.path.join(base_dir, 'models', str(to_gen), filename))
        print(f'_breed_new_model_breed_crossover_parameters() breed_crossover_parameters {model_no_1=} {model_no_2=} to {filename=} {provenance=} {pedigrees=}')
        return filename, provenance, pedigrees

    def _breed_new_model(self, gen: int):
        while True:
            with self.connection:
                self.connection.execute('''BEGIN EXCLUSIVE''')
                cursor = self.connection.cursor()

                model_plan_info = cursor.execute('''
                    SELECT model_plan_no, action, model_no_1, model_no_2 FROM model_plans WHERE gen = ? AND NOT promised AND NOT COMPLETE
                        ORDER BY random()
                        LIMIT 1
                ''', (gen, )).fetchone()

                if model_plan_info:
                    print(f'_breed_new_model() produced new action {model_plan_info=}')
                else:
                    model_plan_info = cursor.execute('''
                        SELECT model_plan_no, action, model_no_1, model_no_2 FROM model_plans WHERE gen = ? AND promised AND NOT COMPLETE
                            ORDER BY random()
                            LIMIT 1
                    ''', (gen, )).fetchone()

                    if model_plan_info:
                        print(f'_breed_new_model() produced previously promised action {model_plan_info=}')

            if not model_plan_info:
                print('_breed_new_model() has nothing to do')
                break  # Nothing to do. All models have been bread.

            (model_plan_no, action, model_no_1, model_no_2) = model_plan_info

            if action == 'copy':
                filename, provenance, pedigrees = self._breed_new_model_copy(model_no_1, gen)
            elif action == 'mutate':
                filename, provenance, pedigrees = self._breed_new_model_mutate(model_no_1, gen)
            elif action == 'breed_crossover_parameters':
                filename, provenance, pedigrees = self._breed_new_model_breed_crossover_parameters(model_no_1, model_no_2, gen)
            else:
                raise NotImplementedError()
            
            with self.connection:
                self.connection.execute('''BEGIN EXCLUSIVE''')
                cursor = self.connection.cursor()

                (complete, ) = cursor.execute('''SELECT complete FROM model_plans WHERE model_plan_no = ?''', (model_plan_no, )).fetchone()

                if complete:
                    print(f'_breed_new_model() discarding generated model due to race condition {model_plan_info=}')
                    self._unbreed_model(gen, filename)
                    break  # Nothing to do. All models have been bread.

                cursor.execute('''UPDATE model_plans SET complete = TRUE WHERE model_plan_no = ?''', (model_plan_no, ))

            return filename, provenance, pedigrees

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
