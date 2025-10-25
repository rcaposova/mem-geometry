import os, random, numpy as np, torch

def set_seed(seed: int = 0, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def ensure_dir(path:str):
    os.makedirs(path, exist_ok=True)