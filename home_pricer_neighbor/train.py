"""kNN regression R&D study on California Housing.

Experiments:
1. 4-D grid sweep: k × weights × metric × scaled. Logs train/test R²
   and MAE.
2. Train vs test R² curve over k (single best metric/weights/scale
   combo) — the classic bias-variance picture.
3. Residual plots for the best and worst k values on a held-out split.

Outputs: ``artifacts/knn_reg_grid.csv``, ``artifacts/r2_vs_k.png``,
``artifacts/residuals_best.png``, ``artifacts/residuals_worst.png``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import save_csv, save_plot, set_seed  # noqa: E402

from data import load_clean  # noqa: E402
from model import build_knn  # noqa: E402


HERE = Path(__file__).resolve().parent
K_VALUES = [1, 3, 5, 10, 20, 50, 100, 200]
WEIGHTS = ["uniform", "distance"]
METRICS = [("euclidean", 2), ("manhattan", 1)]


def grid_sweep(Xtr, Xte, ytr, yte) -> pd.DataFrame:
    print("\n== 4-D grid (k × weights × metric × scaled) ==")
    rows = []
    for k in K_VALUES:
        for w in WEIGHTS:
            for mname, p in METRICS:
                for scaled in (True, False):
                    pipe = build_knn(k, weights=w, metric="minkowski",
                                     p=p, scaled=scaled).fit(Xtr, ytr)
                    ptr = pipe.predict(Xtr); pte = pipe.predict(Xte)
                    rows.append({
                        "k": k, "weights": w, "metric": mname,
                        "scaled": bool(scaled),
                        "train_r2": float(r2_score(ytr, ptr)),
                        "test_r2": float(r2_score(yte, pte)),
                        "test_mae": float(mean_absolute_error(yte, pte)),
                    })
    df = pd.DataFrame(rows)
    print(df.sort_values("test_r2", ascending=False).head(10)
            .to_string(index=False))
    return df


def bias_variance_curve(Xtr, Xte, ytr, yte) -> tuple[list, list, list]:
    print("\n== train vs test R² curve (best non-k config: distance/euclidean/scaled) ==")
    train_r2, test_r2, test_mae = [], [], []
    for k in K_VALUES:
        pipe = build_knn(k, weights="distance", metric="minkowski",
                         p=2, scaled=True).fit(Xtr, ytr)
        ptr = pipe.predict(Xtr); pte = pipe.predict(Xte)
        train_r2.append(r2_score(ytr, ptr))
        test_r2.append(r2_score(yte, pte))
        test_mae.append(mean_absolute_error(yte, pte))
        print(f"  k={k:>4d}  train_R²={train_r2[-1]:.4f}  "
              f"test_R²={test_r2[-1]:.4f}  test_MAE={test_mae[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(K_VALUES, train_r2, "o-", label="train R²")
    ax.plot(K_VALUES, test_r2, "s-", label="test R²")
    ax.set_xscale("log")
    ax.set_xlabel("k (log scale)"); ax.set_ylabel("R²")
    ax.set_title("kNN regression: bias-variance trade-off")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.grid(alpha=0.3); ax.legend()
    save_plot(fig, HERE / "artifacts" / "r2_vs_k.png")
    return train_r2, test_r2, test_mae


def residual_plot(yte, pred, k: int, label: str, out_path: Path) -> None:
    resid = np.asarray(yte) - np.asarray(pred)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(pred, resid, s=8, alpha=0.4)
    axes[0].axhline(0, color="red", linewidth=0.7)
    axes[0].set_xlabel("predicted")
    axes[0].set_ylabel("residual (true − pred)")
    axes[0].set_title(f"Residuals vs predicted — {label} k={k}")
    axes[1].hist(resid, bins=40, alpha=0.7, color="steelblue")
    axes[1].axvline(0, color="red", linewidth=0.7)
    axes[1].set_xlabel("residual")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"Residual distribution — {label} k={k}")
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
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )
    print(f"[data] X={X.shape}  y range=[{y.min():.3f}, {y.max():.3f}]")

    grid_df = grid_sweep(Xtr, Xte, ytr, yte)
    save_csv(grid_df.to_dict(orient="records"), out_dir / "knn_reg_grid.csv")

    _, test_r2, _ = bias_variance_curve(Xtr, Xte, ytr, yte)
    best_k = K_VALUES[int(np.argmax(test_r2))]
    worst_k = K_VALUES[int(np.argmin(test_r2))]
    print(f"\n[viz] best k={best_k}  worst k={worst_k}")

    for k, label, name in ((best_k, "best", "best"), (worst_k, "worst", "worst")):
        pipe = build_knn(k, weights="distance", metric="minkowski",
                         p=2, scaled=True).fit(Xtr, ytr)
        residual_plot(yte, pipe.predict(Xte), k, label,
                      out_dir / f"residuals_{name}.png")


if __name__ == "__main__":
    main()
