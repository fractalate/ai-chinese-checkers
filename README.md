# AI Chinese Checkers

Welcome to my Chinese Checkers AI project: a neural network for playing Chinese Checkers.
It uses PyTorch to train a neural network which can make decisions about moves in a two player game.

The project is in the "validating and adjusting the training methodology phase", which means that we are using small models for expedient testing and that the methodology for training may change, invalidating previously generated models.
Models may be trained, but they may not be effective.

## Setup and Running

```
pip install torch
```

You may train models, which produces data in the `out/basic_trainer/sample` directory by running:

```
python3 train.py
```

Several of these may be run in parallel; for example:

```
python3 train.py &
python3 train.py &
python3 train.py &
```

You may inspect the progress of training and how well each model has performed by looking at the training database:

```
sqlite3 out/basic_trainer/sample/training.db
```

and executing some queries; for example:

```
sqlite> select * from models limit 5;
1|3e60730b-ef76-46aa-8748-696aad10c800.pth|1
2|8b02eab2-ac04-47eb-ba65-22d40d33e8d5.pth|1
3|2a109428-add2-44e7-8276-71b47c6b2648.pth|1
4|5ca90fce-0661-481d-a661-c6ba328f2292.pth|1
5|fb462297-34bc-4af1-8615-aaaeb7aa474f.pth|1
```
