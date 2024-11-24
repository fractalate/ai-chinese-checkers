import torch
import torch.nn as nn

import board

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# l is size of neighborhood around pieces to consider
def calculate_input_neurons(l: int) -> int:
    # For each of the 121 cells on the board:
    # - one input if cell is friendly
    # - one input if cell is enemy
    # For each of the 20 pieces in a 2 player game:
    #   For each of the COUNT cells around each piece
    #   - one input if cell is friendly
    #   - one input if cell is enemy
    #   - one input if cell is impassible
    #   - one input if cell is backtracking
    COUNT = len(board.list_xy_around(0, 0, l))
    return 121 * 2 + 20 * COUNT * 4

N = calculate_input_neurons(6) # is this too big?
H = 100_000 # how big is too big?
M = 121*121 + 1

print(f'{N=} {H=} {M=}')

import sys
sys.exit()

model = NeuralNetwork(input_size=N, output_size=M, hidden_size=H)

for i in range(121):
    print(board.cell_to_xy(i))

'''
model2 = NeuralNetwork(input_size=N, output_size=M, hidden_size=H)

# Example input data
X = torch.randn(5, N)  # 5 samples with N features each

# Generate outputs
outputs = model(X)
print('Outputs:', outputs)
outputs = model(X)
print('Outputs:', outputs)
outputs = model2(X)
print('Outputs:', outputs)
'''
