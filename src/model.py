import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Simple 2-layer MLP for classification.
    input_dim -> hidden_dim -> output_dim
    """
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, return_hidden=False):
        h = F.relu(self.fc1(x)) # hidden layer: compute + activation
        out = self.fc2(h) # output layer: compute logits
        if return_hidden:
            return out, h
        return out
    