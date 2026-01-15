import torch as torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple feedforward neural network for block generation
class BlockGenerationNet(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, output_size=27):
        super(BlockGenerationNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, 64)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x