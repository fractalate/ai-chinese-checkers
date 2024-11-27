# AI Chinese Checkers

Welcome to my Chinese Checkers AI project: a neural network for playing Chinese Checkers.
It uses PyTorch to train a neural network which can make decisions about moves in a two player game.

## Setup

```
pip install torch
```

## Training

```
python3 train.py
```

This will create some initial models if none are present in the `generations` directory.
Models are located at `generations/X/Y.model` where `X` is generation number and `Y` is model number.
Match results are stored in `generations/X/matches.csv`.
