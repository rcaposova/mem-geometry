import numpy as np, random
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=32, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
    def forward(self, x, return_h=False):
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        return (logits, h) if return_h else logits


@torch.no_grad()
def accuracy(model, x, y):
    logits = model(x)
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def main(seed=0, epochs=50, lr=1e-2, hidden=32):
    set_seed(seed)

    # 1) data (Task A prototype we’ll use tomorrow)
    Xa, ya = make_moons(n_samples=1000, noise=0.2, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.3, random_state=seed)

    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.long)
    yte = torch.tensor(yte, dtype=torch.long)

    # 2) model + optimizer
    model = MLP(in_dim=2, hidden=hidden, out_dim=2)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # 3) training loop
    for epoch in range(1, epochs+1):
        model.train()
        logits = model(Xtr)
        loss = F.cross_entropy(logits, ytr)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if epoch % 10 == 0 or epoch ==1:
            model.eval()
            acc_tr = accuracy(model, Xtr, ytr)
            acc_te = accuracy(model, Xte, yte)
            print(f"epoch {epoch:03d} | loss {loss.item():.4f} | acc_tr {acc_tr:.3f} | acc_te {acc_te:.3f}")
    
    # 4) quick check: extract hidden activations for later geometry
    model.eval()
    with torch.no_grad():
        _,H = model(Xte, return_h=True) # [N, hidden]
    print("Hidden activations shape:", H.shape)


if __name__ == "__main__":
    main()