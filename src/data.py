import torch
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split
import numpy as np

def make_task_A(seed=0, n_samples=1000, noise=0.2):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.long)
    yte = torch.tensor(yte, dtype=torch.long)
    return Xtr, Xte, ytr, yte


def make_task_B(seed=0, n_samples=1000):
    # different geometry: two Gaussian blobs, rotated
    X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=1.2, random_state=seed)
    # simple rotation to ensure non-trivial mapping
    theta = np.deg2rad(35)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
    X = (X @ R.T).astype("float32")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    return (torch.tensor(Xtr, dtype=torch.float32),
            torch.tensor(Xte, dtype=torch.float32),
            torch.tensor(ytr, dtype=torch.long),
            torch.tensor(yte, dtype=torch.long))
