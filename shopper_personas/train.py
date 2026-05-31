"""KMeans R&D study on Mall Customers.

Experiments:
1. K-sweep (2..12): inertia, silhouette, Davies-Bouldin,
   Calinski-Harabasz — all four metrics plotted.
2. Init comparison: `k-means++` vs `random` × `n_init ∈ {1, 5, 10, 25}`.
3. `MiniBatchKMeans` vs `KMeans` wall-clock + ARI agreement.
4. 2-D scatter of best K with centroids on multiple feature pairs.

Artifacts:
- ``k_sweep.csv``, ``k_sweep.png`` (2×2 panel)
- ``init_comparison.csv``
- ``minibatch_vs_full.csv``
- ``clusters_<pair>_k<k>.png`` (one per feature pair)
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import save_csv, save_plot, set_seed  # noqa: E402

from data import load_clean  # noqa: E402
from model import build_kmeans  # noqa: E402


HERE = Path(__file__).resolve().parent
K_RANGE = list(range(2, 13))


def k_sweep(X, *, seed) -> pd.DataFrame:
    print("\n== K-sweep with 4 metrics ==")
    Xs = StandardScaler().fit_transform(X)
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
        labels = km.labels_
        sil = silhouette_score(Xs, labels) if k > 1 else float("nan")
        db = davies_bouldin_score(Xs, labels) if k > 1 else float("nan")
        ch = calinski_harabasz_score(Xs, labels) if k > 1 else float("nan")
        rows.append({"k": k,
                     "inertia": float(km.inertia_),
                     "silhouette": float(sil),
                     "davies_bouldin": float(db),
                     "calinski_harabasz": float(ch)})
        print(f"  k={k:>2d}  inertia={km.inertia_:.2f}  sil={sil:.4f}  "
              f"DB={db:.4f}  CH={ch:.2f}")
    return pd.DataFrame(rows)


def plot_k_sweep(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    spec = [
        ("inertia", "Elbow (lower-bend is better)"),
        ("silhouette", "Silhouette (higher is better)"),
        ("davies_bouldin", "Davies-Bouldin (lower is better)"),
        ("calinski_harabasz", "Calinski-Harabasz (higher is better)"),
    ]
    for ax, (col, title) in zip(axes.ravel(), spec):
        ax.plot(df["k"], df[col], "o-")
        ax.set_xlabel("k"); ax.set_ylabel(col)
        ax.set_title(title); ax.grid(alpha=0.3)
    save_plot(fig, out_path)


def init_comparison(X, *, seed, k: int = 5) -> pd.DataFrame:
    print(f"\n== init comparison at k={k} ==")
    Xs = StandardScaler().fit_transform(X)
    rows = []
    for init in ("k-means++", "random"):
        for n_init in (1, 5, 10, 25):
            inertias, sils, times = [], [], []
            for s in range(seed, seed + 5):
                t0 = time.perf_counter()
                km = KMeans(n_clusters=k, init=init, n_init=n_init,
                            random_state=s).fit(Xs)
                times.append(time.perf_counter() - t0)
                inertias.append(km.inertia_)
                sils.append(silhouette_score(Xs, km.labels_))
            rows.append({"init": init, "n_init": int(n_init),
                         "inertia_mean": float(np.mean(inertias)),
                         "inertia_std": float(np.std(inertias)),
                         "silhouette_mean": float(np.mean(sils)),
                         "time_mean_s": float(np.mean(times))})
            print(f"  init={init:>10s}  n_init={n_init:>2d}  "
                  f"inertia={np.mean(inertias):.2f}±{np.std(inertias):.2f}  "
                  f"sil={np.mean(sils):.4f}  t={np.mean(times):.3f}s")
    return pd.DataFrame(rows)


def minibatch_vs_full(X, *, seed) -> pd.DataFrame:
    print("\n== MiniBatchKMeans vs KMeans ==")
    Xs = StandardScaler().fit_transform(X)
    rows = []
    for k in (3, 5, 8):
        t0 = time.perf_counter()
        full = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
        t_full = time.perf_counter() - t0
        t0 = time.perf_counter()
        mini = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=seed,
                               batch_size=64).fit(Xs)
        t_mini = time.perf_counter() - t0
        ari = adjusted_rand_score(full.labels_, mini.labels_)
        rows.append({"k": int(k),
                     "full_inertia": float(full.inertia_),
                     "mini_inertia": float(mini.inertia_),
                     "full_time_s": float(t_full),
                     "mini_time_s": float(t_mini),
                     "ari_agreement": float(ari)})
        print(f"  k={k}  full={t_full:.3f}s  mini={t_mini:.3f}s  "
              f"ARI={ari:.4f}")
    return pd.DataFrame(rows)


def scatter_with_centroids(X, pair, k: int, *, seed, out_path: Path) -> None:
    pipe = build_kmeans(n_clusters=k, random_state=seed).fit(X)
    labels = pipe.named_steps["km"].labels_
    centers = pipe.named_steps["km"].cluster_centers_
    centers_raw = pipe.named_steps["scaler"].inverse_transform(centers)
    f0, f1 = pair
    i0 = list(X.columns).index(f0); i1 = list(X.columns).index(f1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X.iloc[:, i0], X.iloc[:, i1], c=labels, cmap="tab10",
               s=40, alpha=0.75, edgecolor="k")
    ax.scatter(centers_raw[:, i0], centers_raw[:, i1], c="black",
               marker="X", s=220, label="centroids")
    ax.set_xlabel(f0); ax.set_ylabel(f1)
    ax.set_title(f"KMeans clusters  k={k}  features=({f0}, {f1})")
    ax.legend()
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

    X, headline = load_clean()
    print(f"[data] X={X.shape}  headline features={headline}")

    sweep = k_sweep(X, seed=args.seed)
    save_csv(sweep.to_dict(orient="records"), out_dir / "k_sweep.csv")
    plot_k_sweep(sweep, out_dir / "k_sweep.png")

    init_df = init_comparison(X, seed=args.seed)
    save_csv(init_df.to_dict(orient="records"), out_dir / "init_comparison.csv")

    mini_df = minibatch_vs_full(X, seed=args.seed)
    save_csv(mini_df.to_dict(orient="records"),
             out_dir / "minibatch_vs_full.csv")

    # best K by silhouette
    best_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
    print(f"\n[viz] best K by silhouette = {best_k}")
    # scatter for all pairs of the 3 numeric headline-ish columns
    numeric_cols = [c for c in X.columns if c != "Gender_male"]
    pairs = list(itertools.combinations(numeric_cols, 2))
    for pair in pairs:
        out = out_dir / f"clusters_{pair[0]}_vs_{pair[1]}_k{best_k}.png"
        # sanitize filename
        out = out.with_name(out.name.replace(" ", "_").replace("/", "-"))
        scatter_with_centroids(X, pair, best_k, seed=args.seed, out_path=out)


if __name__ == "__main__":
    main()
