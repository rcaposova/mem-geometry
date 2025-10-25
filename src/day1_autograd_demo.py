import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# --- seeding for repeatability ---
def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(0)

# --- 1) make some fake inputs (4 samples, 10 features) ---
x = torch.randn(4, 10)
print("x shape:", x.shape)

# --- 2) define a linear layer and a ReLU ---
layer = nn.Linear(10, 5) # weights shape [5,10], bias [5]
h = F.relu(layer(x))
print("h (after relu) shape:", h.shape)

# --- 3) define a simple scalar loss to backprop ---
loss = h.sum()
print("loss (scalar):", float(loss))

# --- 4) backpropagate ---
layer.zero_grad() # clear old grads if any
loss.backward() # autograd computes d(loss)/d(params)

print("weight.grad shape:", layer.weight.grad.shape)
print("mean grad magnitude:", layer.weight.grad.abs().mean().item())