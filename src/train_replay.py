import os, csv
import torch
import torch.nn.functional as F
from .model import MLP
from .utils import set_seed, ensure_dir
from .data import make_task_A, make_task_B

def replay_buffer(X_old, y_old, X_new, y_new, replay_ratio=0.5):
    n_replay = int(replay_ratio * len(X_new))
    idx = torch.randperm(len(X_old))[:n_replay]
    X_mix = torch.cat([X_new, X_old[idx]], dim=0)
    y_mix = torch.cat([y_new, y_old[idx]], dim=0)
    return X_mix, y_mix

def run_seq_A_then_B_replay(epochs_A=50, epochs_B=50, lr=1e-2, replay_ratio=0.5,
                            seed=0, outdir="experiments/ab_replay"):
    set_seed(seed)
    ensure_dir(outdir)
    print("[replay] outdir:", os.path.abspath(outdir))   

    # ----- Task A -----
    XA_tr, XA_te, yA_tr, yA_te = make_task_A(seed=seed)
    model = MLP(input_dim=2, hidden_dim=32, output_dim=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs_A):
        loss = F.cross_entropy(model(XA_tr), yA_tr)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        accA_before = (model(XA_te).argmax(1) == yA_te).float().mean().item()
        _, HA_before = model(XA_te, return_hidden=True)
    p_before = os.path.join(outdir, "A_hidden_before_B.pt")
    torch.save(HA_before.cpu(), p_before)
    print("[replay] saved:", p_before)                   

    # ----- Task B with Replay -----
    XB_tr, XB_te, yB_tr, yB_te = make_task_B(seed=seed)

    # Smaller LR on B helps preserve A
    opt = torch.optim.Adam(model.parameters(), lr=lr * 0.5)

    model.train()
    for ep in range(1, epochs_B + 1):
        X_mix, y_mix = replay_buffer(XA_tr, yA_tr, XB_tr, yB_tr, replay_ratio)
        loss = F.cross_entropy(model(X_mix), y_mix)
        opt.zero_grad(); loss.backward(); opt.step()

        if ep in (1, 10, epochs_B):
            model.eval()
            with torch.no_grad():
                accA_chk = (model(XA_te).argmax(1) == yA_te).float().mean().item()
                accB_chk = (model(XB_te).argmax(1) == yB_te).float().mean().item()
            print(f"[replay ep {ep:03d}] accA={accA_chk:.3f} accB={accB_chk:.3f}")
            model.train()

    # ----- Final eval & saves -----
    model.eval()
    with torch.no_grad():
        accA_after = (model(XA_te).argmax(1) == yA_te).float().mean().item()
        accB_after = (model(XB_te).argmax(1) == yB_te).float().mean().item()
        _, HA_after = model(XA_te, return_hidden=True)

    p_after = os.path.join(outdir, "A_hidden_after_B.pt")
    torch.save(HA_after.cpu(), p_after)
    print("[replay] saved:", p_after)                    

    csv_path = os.path.join(outdir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["phase","acc_A","acc_B"])
        w.writeheader()
        w.writerow({"phase":"after_A", "acc_A":accA_before, "acc_B":""})
        w.writerow({"phase":"after_B", "acc_A":accA_after, "acc_B":accB_after})
    print("[replay] saved:", csv_path)                   

if __name__ == "__main__":
    run_seq_A_then_B_replay()
