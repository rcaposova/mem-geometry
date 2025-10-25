import torch
import torch.nn.functional as F
from torch import optim
import csv, os

from .utils import set_seed, ensure_dir
from .model import MLP
from .data  import make_task_A, make_task_B

@torch.no_grad()
def accuracy(model, X, y):
    preds = model(X).argmax(1)
    return (preds == y).float().mean().item()


def train_one_task(model, Xtr, ytr, epochs=50, lr=1e-2):
    opt = optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        logits = model(Xtr)
        loss = F.cross_entropy(logits, ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def run_seq_A_then_B(epochs_A=50, epochs_B=50, lr=1e-2, hidden=32, seed=0,
                     outdir="experiments/ab_no_replay"):
    set_seed(seed)
    ensure_dir(outdir)

    # ---------- Task A data ----------
    XA_tr, XA_te, yA_tr, yA_te = make_task_A(seed=seed)

    # ---------- init model ----------
    model = MLP(input_dim=2, hidden_dim=hidden, output_dim=2)

    # ---------- train on A ----------
    model = train_one_task(model, XA_tr, yA_tr, epochs=epochs_A, lr=lr)

    # metrics after A
    accA_afterA = accuracy(model, XA_te, yA_te)

    # save hidden activations on A BEFORE training B
    model.eval()
    with torch.no_grad():
        _, HA_before = model(XA_te, return_hidden=True) # [NA, hidden]
    torch.save(HA_before, os.path.join(outdir, "A_hidden_before_B.pt"))

    # ---------- Task B data ----------
    XB_tr, XB_te, yB_tr, yB_te = make_task_B(seed=seed)

    # ---------- continue training on B (no replay) ----------
    model = train_one_task(model, XB_tr, yB_tr, epochs=epochs_B, lr=lr)

    # metrics after B
    accA_afterB = accuracy(model, XA_te, yA_te) # forgetting shows here
    accB_afterB = accuracy(model, XB_te, yB_te)

    # save hidden activations on A AFTER training B
    model.eval()
    with torch.no_grad():
        _, HA_after = model(XA_te, return_hidden=True)
    torch.save(HA_after, os.path.join(outdir, "A_hidden_after_B.pt"))

    # ---------- write a tiny CSV log ----------
    csv_path = os.path.join(outdir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["phase","acc_A","acc_B"])
        w.writeheader()
        w.writerow({"phase":"after_A", "acc_A":accA_afterA, "acc_B":""})
        w.writerow({"phase":"after_B", "acc_A":accA_afterB, "acc_B":accB_afterB})

        print(f"[done] acc_A after A: {accA_afterA:.3f}")
        print(f"[done] acc_A after B: {accA_afterB:.3f}  (forgetting goes up if this drops)")
        print(f"[done] acc_B after B: {accB_afterB:.3f}")
        print(f"saved: {csv_path}")
        print("saved: A_hidden_before_B.pt, A_hidden_after_B.pt")

if __name__ == "__main__":
    run_seq_A_then_B()
