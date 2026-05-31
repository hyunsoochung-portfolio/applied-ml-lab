"""4-way hyperparameter-search comparison on Adult Income (RandomForest).

The same search space (`n_estimators`, `max_depth`, `min_samples_leaf`,
`max_features`) is fed to:
- `GridSearchCV`
- `RandomizedSearchCV`
- `HalvingGridSearchCV` (sklearn.experimental)
- `HalvingRandomSearchCV`

Outputs:
- ``search_comparison.csv`` — best params, best CV score, wall-clock
- ``pareto.png`` — wall-clock vs best CV score with Pareto front marked
- ``rand_n_iter_sensitivity.csv`` — sweep of `n_iter` for RandomizedSearch
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import pareto_front, save_csv, save_plot, set_seed  # noqa: E402

from data import load_clean  # noqa: E402
from model import build_rf  # noqa: E402


HERE = Path(__file__).resolve().parent

GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 4, 16],
    "max_features": ["sqrt", 0.5],
}

DIST = {
    "n_estimators": randint(100, 400),
    "max_depth": randint(5, 30),
    "min_samples_leaf": randint(1, 32),
    "max_features": uniform(0.3, 0.6),
}


def baseline_cv(X, y, *, seed) -> None:
    print("\n== baseline cross_validate (defaults) ==")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    res = cross_validate(build_rf(n_estimators=100), X, y, cv=cv,
                         scoring=["accuracy", "roc_auc"], n_jobs=-1)
    for k in ("test_accuracy", "test_roc_auc"):
        print(f"  {k:>14s}: {res[k].mean():.4f} ± {res[k].std():.4f}")


def run_searches(Xs, ys, *, seed) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    rows = []
    for name, search in (
        ("GridSearchCV", GridSearchCV(
            build_rf(), GRID, scoring="roc_auc", cv=cv, n_jobs=-1)),
        ("RandomizedSearchCV", RandomizedSearchCV(
            build_rf(), DIST, n_iter=20, scoring="roc_auc",
            cv=cv, n_jobs=-1, random_state=seed)),
        ("HalvingGridSearchCV", HalvingGridSearchCV(
            build_rf(), GRID, scoring="roc_auc", cv=cv,
            n_jobs=-1, factor=3, random_state=seed)),
        ("HalvingRandomSearchCV", HalvingRandomSearchCV(
            build_rf(), DIST, n_candidates=40, scoring="roc_auc",
            cv=cv, n_jobs=-1, factor=3, random_state=seed)),
    ):
        print(f"\n== {name} ==")
        t0 = time.perf_counter()
        search.fit(Xs, ys)
        dt = time.perf_counter() - t0
        rows.append({
            "method": name,
            "best_score": float(search.best_score_),
            "wall_clock_s": float(dt),
            "best_params": str(search.best_params_),
        })
        print(f"  best_score={search.best_score_:.4f}  "
              f"time={dt:.1f}s  best_params={search.best_params_}")
    return pd.DataFrame(rows)


def plot_pareto(df: pd.DataFrame, out_path: Path) -> None:
    pts = list(zip(df["wall_clock_s"].tolist(),
                   df["best_score"].tolist()))
    front_idx = set(pareto_front(pts, minimize_x=True, maximize_y=True))
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, row in df.iterrows():
        colour = "C3" if i in front_idx else "C0"
        ax.scatter(row["wall_clock_s"], row["best_score"],
                   s=120, color=colour)
        ax.annotate(row["method"],
                    (row["wall_clock_s"], row["best_score"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("wall-clock seconds")
    ax.set_ylabel("best CV ROC-AUC")
    ax.set_title("Search Pareto front (red = non-dominated)")
    ax.grid(alpha=0.3)
    save_plot(fig, out_path)


def rand_n_iter_sensitivity(Xs, ys, *, seed) -> pd.DataFrame:
    print("\n== RandomizedSearchCV n_iter sensitivity ==")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    rows = []
    for n_iter in (10, 50, 200):
        rs = RandomizedSearchCV(build_rf(), DIST, n_iter=n_iter,
                                scoring="roc_auc", cv=cv,
                                n_jobs=-1, random_state=seed)
        t0 = time.perf_counter()
        rs.fit(Xs, ys)
        dt = time.perf_counter() - t0
        rows.append({"n_iter": n_iter,
                     "best_score": float(rs.best_score_),
                     "wall_clock_s": float(dt),
                     "best_params": str(rs.best_params_)})
        print(f"  n_iter={n_iter:>3d}  best={rs.best_score_:.4f}  "
              f"time={dt:.1f}s")
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--subsample", type=int, default=15000,
                   help="rows from Adult Income to feed the searches")
    p.add_argument("--output-dir", type=str, default=str(HERE / "artifacts"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_clean()
    print(f"[data] X={X.shape}  positive rate={y.mean():.3f}")
    baseline_cv(X, y, seed=args.seed)

    Xs, _, ys, _ = train_test_split(
        X, y, train_size=args.subsample, stratify=y, random_state=args.seed
    )
    print(f"\n[search] using subsample of {len(Xs)} rows")

    df = run_searches(Xs, ys, seed=args.seed)
    save_csv(df.to_dict(orient="records"), out_dir / "search_comparison.csv")
    plot_pareto(df, out_dir / "pareto.png")

    sens = rand_n_iter_sensitivity(Xs, ys, seed=args.seed)
    save_csv(sens.to_dict(orient="records"),
             out_dir / "rand_n_iter_sensitivity.csv")


if __name__ == "__main__":
    main()
