"""PCA R&D study on Breast Cancer Wisconsin.

Experiments:
1. PCA vs KernelPCA (linear / rbf / poly) — downstream LogisticRegression
   accuracy on a fixed split.
2. Scree + cumulative explained-variance plots.
3. n_components sweep (1..30) — downstream LogReg accuracy curve.
4. Reconstruction error per n_components (Frobenius norm of
   `X - X_reconstructed` on standardised features).
5. Whitening on/off ablation.
6. 2-D + 3-D PCA scatter coloured by class.

Artifacts: ``pca_vs_kpca.csv``, ``variance.png``,
``components_sweep.csv``, ``components_sweep.png``,
``reconstruction.csv``, ``reconstruction.png``,
``whitening_ablation.csv``, ``pc_scatter_2d.png``, ``pc_scatter_3d.png``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D projection hook)
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import save_csv, save_plot, set_seed  # noqa: E402

from data import load_clean  # noqa: E402
from model import build_kpca_logreg, build_pca_logreg  # noqa: E402


HERE = Path(__file__).resolve().parent


def pca_vs_kpca(Xtr, Xte, ytr, yte, *, seed) -> pd.DataFrame:
    print("\n== PCA vs KernelPCA downstream accuracy ==")
    rows = []
    pipe = build_pca_logreg(10).fit(Xtr, ytr)
    rows.append({"method": "PCA", "kernel": "",
                 "test_acc": float(accuracy_score(yte, pipe.predict(Xte)))})
    print(f"  PCA(linear)     acc={rows[-1]['test_acc']:.4f}")
    for kernel in ("linear", "rbf", "poly"):
        try:
            pipe = build_kpca_logreg(10, kernel=kernel).fit(Xtr, ytr)
            acc = float(accuracy_score(yte, pipe.predict(Xte)))
        except Exception as e:
            print(f"  KernelPCA({kernel}) failed: {e!r}")
            acc = float("nan")
        rows.append({"method": "KernelPCA", "kernel": kernel,
                     "test_acc": acc})
        print(f"  KernelPCA({kernel:>6s}) acc={acc:.4f}")
    return pd.DataFrame(rows)


def variance_curve(Xs, out_path: Path) -> pd.DataFrame:
    pca = PCA(random_state=42).fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.arange(1, len(cum) + 1)
    ax.bar(xs, pca.explained_variance_ratio_, alpha=0.6, label="per-PC ratio")
    ax.plot(xs, cum, "ro-", label="cumulative")
    ax.axhline(0.9, color="grey", linestyle=":", label="0.90 threshold")
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance ratio")
    ax.set_title("Scree + cumulative explained variance")
    ax.legend(); ax.grid(alpha=0.3)
    save_plot(fig, out_path)
    return pd.DataFrame({
        "pc": xs,
        "explained_ratio": pca.explained_variance_ratio_,
        "cumulative": cum,
    })


def components_sweep(X, y, *, seed, ns=None) -> pd.DataFrame:
    print("\n== n_components sweep (1..30) ==")
    if ns is None:
        ns = list(range(1, min(31, X.shape[1] + 1)))
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed,
    )
    rows = []
    for n in ns:
        pipe = build_pca_logreg(n).fit(Xtr, ytr)
        acc = float(accuracy_score(yte, pipe.predict(Xte)))
        rows.append({"n_components": int(n), "test_acc": acc})
        print(f"  n_components={n:>2d}  acc={acc:.4f}")
    return pd.DataFrame(rows)


def plot_components(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["n_components"], df["test_acc"], "o-")
    ax.set_xlabel("n_components"); ax.set_ylabel("test accuracy")
    ax.set_title("LogReg accuracy on PCA-reduced features")
    ax.grid(alpha=0.3)
    save_plot(fig, out_path)


def reconstruction(Xs, *, ns=(1, 2, 5, 10, 15, 20, 30)) -> pd.DataFrame:
    print("\n== reconstruction (Frobenius norm) ==")
    rows = []
    for n in ns:
        if n > Xs.shape[1]:
            continue
        pca = PCA(n_components=int(n), random_state=42).fit(Xs)
        Xr = pca.inverse_transform(pca.transform(Xs))
        frob = float(np.linalg.norm(Xs - Xr))
        cum = float(np.sum(pca.explained_variance_ratio_))
        rows.append({"n_components": int(n),
                     "frob_error": frob,
                     "cumulative_variance": cum})
        print(f"  n_components={n:>2d}  frob_err={frob:.4f}  cum_var={cum:.4f}")
    return pd.DataFrame(rows)


def plot_reconstruction(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(df["n_components"], df["frob_error"], "o-", color="C0",
             label="‖X − X̂‖_F")
    ax1.set_xlabel("n_components")
    ax1.set_ylabel("Frobenius reconstruction error", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(df["n_components"], df["cumulative_variance"], "s--",
             color="C1", label="cum. variance")
    ax2.set_ylabel("cumulative variance ratio", color="C1")
    ax1.set_title("Reconstruction error vs n_components")
    ax1.grid(alpha=0.3)
    save_plot(fig, out_path)


def whitening_ablation(X, y, *, seed) -> pd.DataFrame:
    print("\n== whitening ablation ==")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed,
    )
    rows = []
    for whiten in (False, True):
        for n in (2, 5, 10, 20):
            pipe = build_pca_logreg(n, whiten=whiten).fit(Xtr, ytr)
            acc = float(accuracy_score(yte, pipe.predict(Xte)))
            rows.append({"whiten": bool(whiten),
                         "n_components": int(n),
                         "test_acc": acc})
            print(f"  whiten={str(whiten):>5s}  n={n:>2d}  acc={acc:.4f}")
    return pd.DataFrame(rows)


def pc_scatter_2d(Xs, y, out_path: Path) -> None:
    pcs = PCA(n_components=2, random_state=42).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, marker in zip([0, 1], ["o", "s"]):
        m = y == cls
        label = "malignant" if cls == 0 else "benign"
        ax.scatter(pcs[m, 0], pcs[m, 1], marker=marker, s=30,
                   alpha=0.7, edgecolor="k", label=label)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("First two principal components"); ax.legend()
    save_plot(fig, out_path)


def pc_scatter_3d(Xs, y, out_path: Path) -> None:
    pcs = PCA(n_components=3, random_state=42).fit_transform(Xs)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for cls, marker in zip([0, 1], ["o", "s"]):
        m = y == cls
        label = "malignant" if cls == 0 else "benign"
        ax.scatter(pcs[m, 0], pcs[m, 1], pcs[m, 2], marker=marker,
                   s=18, alpha=0.7, edgecolor="k", label=label)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title("First three principal components"); ax.legend()
    save_plot(fig, out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--output-dir", type=str, default=str(HERE / "artifacts"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_clean()
    print(f"[data] X={X.shape}")
    Xs = StandardScaler().fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=args.seed,
    )
    cmp_df = pca_vs_kpca(Xtr, Xte, ytr, yte, seed=args.seed)
    save_csv(cmp_df.to_dict(orient="records"), out_dir / "pca_vs_kpca.csv")

    var_df = variance_curve(Xs, out_dir / "variance.png")
    save_csv(var_df.to_dict(orient="records"), out_dir / "variance.csv")

    sweep_df = components_sweep(X, y, seed=args.seed)
    save_csv(sweep_df.to_dict(orient="records"),
             out_dir / "components_sweep.csv")
    plot_components(sweep_df, out_dir / "components_sweep.png")

    rec_df = reconstruction(Xs)
    save_csv(rec_df.to_dict(orient="records"), out_dir / "reconstruction.csv")
    plot_reconstruction(rec_df, out_dir / "reconstruction.png")

    wh_df = whitening_ablation(X, y, seed=args.seed)
    save_csv(wh_df.to_dict(orient="records"),
             out_dir / "whitening_ablation.csv")

    pc_scatter_2d(Xs, y.to_numpy(), out_dir / "pc_scatter_2d.png")
    pc_scatter_3d(Xs, y.to_numpy(), out_dir / "pc_scatter_3d.png")


if __name__ == "__main__":
    main()
