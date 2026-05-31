"""Train kNN classifiers on Heart Disease — full R&D sweep.

Headline outputs:
- ``artifacts/knn_sweep.csv``: n_neighbors x weights x metric grid,
  5-seed StratifiedKFold mean ± std on accuracy and F1
- ``artifacts/knn_sweep_heatmap.png``: heatmap of mean accuracy per
  (n_neighbors, weights+metric) cell
- ``artifacts/best_config_bootstrap.csv``: 95% bootstrap CI for the best
  CV config on a held-out test split
- ``artifacts/boundary_<pair>_k<k>.png``: decision boundaries on 3
  different 2-feature pairs (k=1, 5, 25 each)

Also prints a short per-k accuracy table for the headline result so the
console output stays self-contained.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import (  # noqa: E402
    artifacts_dir,
    save_csv,
    save_plot,
    set_seed,
)

from data import load_clean  # noqa: E402
from model import build_knn  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_K_VALUES = [1, 3, 5, 7, 11, 15, 21, 31]
WEIGHTS = ["uniform", "distance"]
# (name, kwargs) — minkowski(p=3) is parameterised via the `p` arg
METRICS = [
    ("euclidean", {"metric": "euclidean"}),
    ("manhattan", {"metric": "manhattan"}),
    ("minkowski_p3", {"metric": "minkowski", "p": 3}),
]
BOUNDARY_PAIRS = [
    ("thalach", "oldpeak"),
    ("age", "chol"),
    ("trestbps", "thalach"),
]


def _pipe(*, k: int, weights: str, metric_kwargs: dict) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights=weights, **metric_kwargs)),
    ])


def cv_score(
    X, y, *, k: int, weights: str, metric_name: str, metric_kwargs: dict, seed: int
):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for tr, te in cv.split(X, y):
        pipe = _pipe(k=k, weights=weights, metric_kwargs=metric_kwargs)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        accs.append(accuracy_score(y.iloc[te], pred))
        f1s.append(f1_score(y.iloc[te], pred))
    return float(np.mean(accs)), float(np.mean(f1s))


def run_grid(X, y, *, k_values, seeds) -> pd.DataFrame:
    rows = []
    for k in k_values:
        for w in WEIGHTS:
            for mname, mkw in METRICS:
                accs, f1s = [], []
                for s in seeds:
                    a, f1 = cv_score(
                        X, y, k=k, weights=w,
                        metric_name=mname, metric_kwargs=mkw, seed=int(s),
                    )
                    accs.append(a); f1s.append(f1)
                rows.append({
                    "n_neighbors": k, "weights": w, "metric": mname,
                    "acc_mean": float(np.mean(accs)),
                    "acc_std": float(np.std(accs)),
                    "f1_mean": float(np.mean(f1s)),
                    "f1_std": float(np.std(f1s)),
                    "n_seeds": len(seeds),
                })
    return pd.DataFrame(rows)


def bootstrap_ci(X, y, *, k: int, weights: str, metric_kwargs: dict,
                 n_boot: int = 200, seed: int = 0) -> dict:
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    pipe = _pipe(k=k, weights=weights, metric_kwargs=metric_kwargs)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    yte_arr = np.asarray(yte)
    rng = np.random.default_rng(seed)
    n = len(yte_arr)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(accuracy_score(yte_arr[idx], pred[idx]))
    boots = np.asarray(boots)
    return {
        "k": k, "weights": weights,
        "test_acc": float(accuracy_score(yte_arr, pred)),
        "boot_mean": float(boots.mean()),
        "boot_ci_lo": float(np.percentile(boots, 2.5)),
        "boot_ci_hi": float(np.percentile(boots, 97.5)),
        "n_boot": n_boot,
    }


def heatmap(df: pd.DataFrame, out_path: Path) -> None:
    df = df.copy()
    df["col"] = df["weights"] + "/" + df["metric"]
    pivot = df.pivot(index="n_neighbors", columns="col", values="acc_mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("(weights / metric)")
    ax.set_ylabel("n_neighbors")
    ax.set_title("kNN: mean CV accuracy (5-seed StratifiedKFold)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}",
                    ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax)
    save_plot(fig, out_path)


def plot_decision_boundary(X, y, pair, k, out_path: Path) -> None:
    Xp = X[list(pair)].to_numpy()
    scaler = StandardScaler().fit(Xp)
    Xs = scaler.transform(Xp)
    model = KNeighborsClassifier(n_neighbors=k).fit(Xs, y)
    pad = 0.5
    x0_min, x0_max = Xs[:, 0].min() - pad, Xs[:, 0].max() + pad
    x1_min, x1_max = Xs[:, 1].min() - pad, Xs[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x0_min, x0_max, 220),
        np.linspace(x1_min, x1_max, 220),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm",
                levels=[-0.5, 0.5, 1.5])
    ax.scatter(Xs[:, 0], Xs[:, 1], c=y, cmap="coolwarm",
               edgecolor="k", s=28, linewidth=0.4)
    ax.set_xlabel(f"{pair[0]} (standardised)")
    ax.set_ylabel(f"{pair[1]} (standardised)")
    ax.set_title(f"kNN decision boundary  k={k}  features={pair}")
    save_plot(fig, out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--output-dir", type=str, default=str(HERE / "artifacts"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed, args.seed + args.n_seeds))

    X, y = load_clean()
    print(f"[data] X={X.shape}  positive rate={y.mean():.3f}")

    print(f"\n== full grid CV sweep ({len(seeds)} seeds) ==")
    grid_df = run_grid(X, y, k_values=DEFAULT_K_VALUES, seeds=seeds)
    save_csv(grid_df.to_dict(orient="records"), out_dir / "knn_sweep.csv")
    print(grid_df.sort_values("acc_mean", ascending=False).head(10)
          .to_string(index=False))

    heatmap(grid_df, out_dir / "knn_sweep_heatmap.png")

    print("\n== quick per-k accuracy on a held-out split (single seed) ==")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=args.seed
    )
    for k in DEFAULT_K_VALUES:
        pipe = build_knn(n_neighbors=k).fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        acc = accuracy_score(yte, pred)
        cm = confusion_matrix(yte, pred)
        print(f"k={k:>3d}  acc={acc:.4f}  confusion={cm.tolist()}")

    best = grid_df.sort_values("acc_mean", ascending=False).iloc[0]
    mkw = dict(METRICS)[best["metric"]]
    print(f"\n== bootstrap 95% CI for best config "
          f"(k={best['n_neighbors']}, w={best['weights']}, m={best['metric']}) ==")
    ci = bootstrap_ci(
        X, y, k=int(best["n_neighbors"]),
        weights=str(best["weights"]),
        metric_kwargs=mkw, seed=args.seed,
    )
    save_csv([ci], out_dir / "best_config_bootstrap.csv")
    print(f"  test_acc={ci['test_acc']:.4f}  "
          f"boot_mean={ci['boot_mean']:.4f}  "
          f"95% CI=[{ci['boot_ci_lo']:.4f}, {ci['boot_ci_hi']:.4f}]")

    print("\n== decision boundary plots ==")
    for pair in BOUNDARY_PAIRS:
        for k in (1, 5, 25):
            plot_decision_boundary(
                Xtr, ytr, pair, k,
                out_dir / f"boundary_{pair[0]}-{pair[1]}_k{k}.png",
            )


if __name__ == "__main__":
    main()
