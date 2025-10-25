import os, csv, math
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .utils import ensure_dir, set_seed
from .data import make_task_A

def load_hidden(indir):
    H_before = torch.load(os.path.join(indir, "A_hidden_before_B.pt"), map_location="cpu")
    H_after  = torch.load(os.path.join(indir, "A_hidden_after_B.pt"),  map_location="cpu")
    return H_before.detach().cpu().numpy(), H_after.detach().cpu().numpy()

def read_metrics(indir):
    csv_path = os.path.join(indir, "metrics.csv")
    acc_A_afterA = acc_A_afterB = acc_B_afterB = None
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["phase"] == "after_A":
                acc_A_afterA = float(row["acc_A"])
            elif row["phase"] == "after_B":
                acc_A_afterB = float(row["acc_A"])
                if row.get("acc_B"):
                    acc_B_afterB = float(row["acc_B"])
    return acc_A_afterA, acc_A_afterB, acc_B_afterB

def pca_plot(H_before, H_after, yA_te, out_png):
    # fit PCA on concatenated data for a shared 2-D basis
    X = np.vstack([H_before, H_after])
    pca = PCA(n_components=2)
    pca.fit(X)
    Zb = pca.transform(H_before)
    Za = pca.transform(H_after)

    plt.figure(figsize=(7,6))
    for cls, clr in [(0, None), (1, None)]:
        idx = (yA_te == cls)
        plt.scatter(Zb[idx,0], Zb[idx,1], marker='o',  alpha=0.45, label=f"class {cls} (before)")
        plt.scatter(Za[idx,0], Za[idx,1], marker='x',  alpha=0.75, label=f"class {cls} (after)")
    var = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    plt.title("Hidden activations on Task A — before vs after learning B")
    plt.legend(loc="best", fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def principal_angles_cos(U, V):
    """
    U, V: orthonormal bases (d x k). Returns cosines of principal angles (k,).
    """
    # SVD of U^T V -> singular values are cos(theta_i)
    M = U.T @ V
    s = np.linalg.svd(M, compute_uv=False)
    return np.clip(s, 0.0, 1.0)

def subspace_overlap(H_before, H_after, k=5):
    # center
    Hb = H_before - H_before.mean(0, keepdims=True)
    Ha = H_after  - H_after.mean(0, keepdims=True)
    # get top-k PCs as orthonormal bases
    Ub, _ , _ = np.linalg.svd(Hb, full_matrices=False)
    Ua, _ , _ = np.linalg.svd(Ha, full_matrices=False)
    Ub = Ub[:, :k]  # (N x k) but bases should be in feature space; use right singular vectors via PCA:
    # redo via PCA to get feature-space bases (components_ are in d-dim)
    pca_b = PCA(n_components=k).fit(Hb)
    pca_a = PCA(n_components=k).fit(Ha)
    B = pca_b.components_.T   # d x k
    A = pca_a.components_.T   # d x k
    cosines = principal_angles_cos(B, A)  # k cosines in [0,1]
    return cosines

def bar_forgetting(accA_A, accA_B, out_png):
    plt.figure(figsize=(5,4))
    plt.bar([0,1], [accA_A, accA_B])
    plt.xticks([0,1], ["A after A", "A after B"])
    plt.ylim(0,1)
    plt.ylabel("Accuracy")
    plt.title(f"Forgetting = {accA_A - accA_B:+.3f}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main(indir, seed=0, k=5, outdir_figs=None):
    set_seed(seed)
    if outdir_figs is None:
        outdir_figs = os.path.join(indir, "figures")
    ensure_dir(outdir_figs)

    # load activations & labels for coloring
    H_before, H_after = load_hidden(indir)
    _, Xte, _, yte = make_task_A(seed=seed)
    yA_te = yte.cpu().numpy()

    # PCA scatter
    geom_png = os.path.join(outdir_figs, "geometry_pca.png")
    pca_plot(H_before, H_after, yA_te, geom_png)

    # subspace overlap (top-k PC subspaces)
    cosines = subspace_overlap(H_before, H_after, k=k)

    # forgetting bar
    accA_A, accA_B, accB_B = read_metrics(indir)
    bar_png = os.path.join(outdir_figs, "forgetting_bar.png")
    bar_forgetting(accA_A, accA_B, bar_png)

    # text summary
    summary = os.path.join(outdir_figs, "summary.txt")
    with open(summary, "w") as f:
        f.write(f"indir: {indir}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"acc_A after A: {accA_A:.3f}\n")
        f.write(f"acc_A after B: {accA_B:.3f}\n")
        if accB_B is not None:
            f.write(f"acc_B after B: {accB_B:.3f}\n")
        f.write(f"forgetting (A): {accA_A - accA_B:+.3f}\n")
        f.write(f"subspace overlap (top-{k} PCs):\n")
        f.write("  cosines: " + ", ".join(f"{c:.3f}" for c in cosines) + "\n")
        f.write(f"  mean cosine: {cosines.mean():.3f}\n")

    print(f"[saved] {geom_png}")
    print(f"[saved] {bar_png}")
    print(f"[saved] {summary}")
    print(f"mean subspace cosine (top-{k} PCs): {cosines.mean():.3f}")

if __name__ == "__main__":
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("--indir", type=str, default="experiments/ab_no_replay")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()
    main(args.indir, seed=args.seed, k=args.k)
