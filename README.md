# AI Chinese Checkers

Welcome to my Chinese Checkers AI project: a neural network for playing Chinese Checkers.
It uses PyTorch to train a neural network which can make decisions about moves in a two player game.

## Setup and Running

Create a virtual environment for use with this project (we will be installing local, editable packages with `pip`):

```bash
python3 -m venv --prompt ai-chinese-checkers venv
```

Activate it with:

```bash
source ./venv/bin/activate
```

Install the dependencies:

```bash
pip install --editable .
```

Run some sample programs:

```bash
python3 -m trainer.bin.play_match
```
