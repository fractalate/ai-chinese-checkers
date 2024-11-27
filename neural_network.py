import torch
import torch.nn as nn

import board

# l is size of neighborhood around pieces to consider
def calculate_input_neurons(l: int) -> int:
    # For each of the 121 cells on the board:
    # - one input if cell is friendly
    # - one input if cell is enemy
    # - one input if cell is backtracking
    # For each of the 20 pieces in a 2 player game:
    #   For each of the COUNT cells around each piece
    #   - one input if cell is friendly
    #   - one input if cell is enemy
    #   - one input if cell is impassible
    #   - one input if cell is backtracking
    #
    # Layout:
    #   [friendly] * 121
    #   [enemy] * 121
    #   [backtracking] * 121
    #   [
    #     [
    #       friendly
    #       enemy
    #       impassable
    #       backtracking
    #     ] * COUNT
    #   ] * 20
    COUNT = len(board.list_xy_around(0, 0, l))
    return 121 * 3 + 4 * 20 * COUNT

L = 6 # is this too big?
N = calculate_input_neurons(L)
H = 100_000 # how big is too big?
M = 121*121 + 1

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(N, H)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(H, M)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def prepare_input_vector(self, game: Game)
