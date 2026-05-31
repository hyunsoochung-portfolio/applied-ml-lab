"""Feature-scaling R&D study on Wholesale Customers.

Three experiments:
1. Scaler × algorithm matrix (`{None, Standard, MinMax, Robust, MaxAbs}`
   × `{kNN, LinearSVC, LogReg, RandomForest}`) — `StratifiedKFold(5)`
   mean ± std, plus wall-clock fit + predict time per cell.
2. Outlier-injection ablation: inject 5% / 10% / 20% large outliers into
   the training rows and re-score StandardScaler vs RobustScaler across
   all algorithms.
3. Headline kNN per-scaler check + flips-vs-no-scaling demo (kept from
   the original module for the textbook view).

Outputs land in ``artifacts/``: ``scaler_algo_matrix.csv``,
``scaler_algo_heatmap.png``, ``outlier_ablation.csv``,
``outlier_ablation.png``, ``timings.csv``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import save_csv, save_plot, set_seed  # noqa: E402

from data import load_clean  # noqa: E402
from model import algorithms, build, build_for, scalers  # noqa: E402


HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1) Scaler × algorithm matrix
# ---------------------------------------------------------------------------

def scaler_algo_matrix(X, y, *, seeds) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n== scaler × algorithm matrix ==")
    rows, timing_rows = [], []
    for algo_name in algorithms().keys():
        for scaler_name in scalers().keys():
            accs = []
            t_fit_total = 0.0; t_pred_total = 0.0
            for s in seeds:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
                for tr, te in cv.split(X, y):
                    pipe = build_for(scaler_name, algo_name, seed=s)
                    t0 = time.perf_counter()
                    pipe.fit(X.iloc[tr], y.iloc[tr])
                    t_fit_total += time.perf_counter() - t0
                    t1 = time.perf_counter()
                    pred = pipe.predict(X.iloc[te])
                    t_pred_total += time.perf_counter() - t1
                    accs.append(accuracy_score(y.iloc[te], pred))
            row = {
                "algorithm": algo_name,
                "scaler": scaler_name,
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs)),
                "n_runs": len(accs),
            }
            rows.append(row)
            timing_rows.append({
                "algorithm": algo_name, "scaler": scaler_name,
                "fit_time_total_s": float(t_fit_total),
                "predict_time_total_s": float(t_pred_total),
                "fit_time_per_fit_s": float(t_fit_total / max(1, len(accs))),
            })
            print(f"  {algo_name:>14s} × {scaler_name:>9s}  "
                  f"acc={row['acc_mean']:.4f} ± {row['acc_std']:.4f}  "
                  f"fit={t_fit_total:.2f}s")
    return pd.DataFrame(rows), pd.DataFrame(timing_rows)


def heatmap_matrix(df: pd.DataFrame, out_path: Path) -> None:
    pivot = df.pivot(index="algorithm", columns="scaler", values="acc_mean")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}",
                    ha="center", va="center", color="white", fontsize=8)
    ax.set_title("Scaler × algorithm — mean CV accuracy")
    fig.colorbar(im, ax=ax)
    save_plot(fig, out_path)


# ---------------------------------------------------------------------------
# 2) Outlier injection ablation
# ---------------------------------------------------------------------------

def _inject_outliers(X_train: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X_out = X_train.copy().to_numpy()
    n = X_out.shape[0]
    k = int(n * frac)
    if k == 0:
        return pd.DataFrame(X_out, columns=X_train.columns)
    idx = rng.choice(n, size=k, replace=False)
    # multiply chosen rows by 50× on a random subset of columns
    for i in idx:
        ncols = rng.integers(1, X_train.shape[1] + 1)
        cols = rng.choice(X_train.shape[1], size=ncols, replace=False)
        X_out[i, cols] *= 50.0
    return pd.DataFrame(X_out, columns=X_train.columns)


def outlier_ablation(X, y, *, seeds, fracs=(0.0, 0.05, 0.10, 0.20)) -> pd.DataFrame:
    print("\n== outlier injection ablation (StandardScaler vs RobustScaler) ==")
    rows = []
    for s in seeds:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=s
        )
        for frac in fracs:
            Xtr_inj = _inject_outliers(Xtr, frac, seed=s)
            for scaler_name in ("standard", "robust"):
                for algo_name in algorithms().keys():
                    pipe = build_for(scaler_name, algo_name, seed=s)
                    pipe.fit(Xtr_inj, ytr)
                    acc = accuracy_score(yte, pipe.predict(Xte))
                    rows.append({
                        "seed": int(s), "outlier_frac": float(frac),
                        "scaler": scaler_name, "algorithm": algo_name,
                        "test_acc": float(acc),
                    })
        print(f"  seed={s}: ablation rows so far={len(rows)}")
    df = pd.DataFrame(rows)
    summary = (df.groupby(["algorithm", "scaler", "outlier_frac"])["test_acc"]
                 .mean().unstack(["scaler", "outlier_frac"]).round(4))
    print(summary.to_string())
    return df


def plot_outlier(df: pd.DataFrame, out_path: Path) -> None:
    agg = (df.groupby(["algorithm", "scaler", "outlier_frac"])["test_acc"]
             .mean().reset_index())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for ax, algo in zip(axes.ravel(), agg["algorithm"].unique()):
        sub = agg[agg["algorithm"] == algo]
        for scaler_name, marker in (("standard", "o"), ("robust", "s")):
            s = sub[sub["scaler"] == scaler_name].sort_values("outlier_frac")
            ax.plot(s["outlier_frac"], s["test_acc"], marker=marker,
                    label=scaler_name)
        ax.set_title(algo)
        ax.set_xlabel("outlier fraction injected")
        ax.set_ylabel("test accuracy")
        ax.grid(alpha=0.3)
        ax.legend()
    save_plot(fig, out_path)


# ---------------------------------------------------------------------------
# 3) Original kNN flip demo (retained)
# ---------------------------------------------------------------------------

def knn_flip_demo(X, y, *, seed: int, k: int = 5) -> None:
    print(f"\n== kNN flip demo (k={k}) ==")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    preds = {}
    for name, scaler in scalers().items():
        pipe = build(scaler, k=k).fit(Xtr, ytr)
        preds[name] = pipe.predict(Xte)
        print(f"  {name:>9s}: acc={accuracy_score(yte, preds[name]):.4f}")
    diff = np.where(np.any(
        np.stack(list(preds.values())) != preds["none"], axis=0))[0]
    if diff.size:
        i = int(diff[0])
        print(f"  first scaler-dependent prediction at row {i} (true={int(yte.iloc[i])}):")
        for n, p in preds.items():
            print(f"    {n:>9s} -> {int(p[i])}")


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

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
    print("[data] raw feature std range: "
          f"{X.std().min():.1f} .. {X.std().max():.1f}")

    mat_df, tim_df = scaler_algo_matrix(X, y, seeds=seeds)
    save_csv(mat_df.to_dict(orient="records"), out_dir / "scaler_algo_matrix.csv")
    save_csv(tim_df.to_dict(orient="records"), out_dir / "timings.csv")
    heatmap_matrix(mat_df, out_dir / "scaler_algo_heatmap.png")

    out_df = outlier_ablation(X, y, seeds=seeds)
    save_csv(out_df.to_dict(orient="records"), out_dir / "outlier_ablation.csv")
    plot_outlier(out_df, out_dir / "outlier_ablation.png")

    knn_flip_demo(X, y, seed=args.seed)


if __name__ == "__main__":
    main()
