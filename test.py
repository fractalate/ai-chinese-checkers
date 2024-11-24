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

N = 10_000 # maybe something like this
H = 100_000 # how big is too big?
M = 121*121 + 1

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
