"""Regularised-regression R&D study on California Housing (degree-2 poly).

Experiments:
1. Ridge α path on a log grid (1e-3 .. 1e3, 13 points) with 5-fold CV R².
2. Lasso α path on the same grid, with coefficient-path plot.
3. ElasticNet (l1_ratio × α) grid, 5 × 13.
4. Held-out test R² comparison of OLS / Ridge / Lasso / ElasticNet at
   each model's CV-best α.

Outputs land in ``artifacts/``: ``ridge_path.csv``, ``lasso_path.csv``,
``lasso_coeff_path.png``, ``enet_grid.csv``,
``model_comparison.csv``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import save_csv, save_plot, set_seed  # noqa: E402

from data import load_clean  # noqa: E402
from model import build_enet, build_lasso, build_linear, build_ridge  # noqa: E402


HERE = Path(__file__).resolve().parent
ALPHA_GRID = np.logspace(-3, 3, 13)
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
POLY_DEGREE = 2


def ridge_path(X, y, *, seed) -> pd.DataFrame:
    print("\n== Ridge α-path (5-fold CV) ==")
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for a in ALPHA_GRID:
        sc = cross_val_score(build_ridge(a, POLY_DEGREE), X, y,
                             cv=cv, scoring="r2", n_jobs=1)
        rows.append({"alpha": float(a),
                     "cv_r2_mean": float(sc.mean()),
                     "cv_r2_std": float(sc.std())})
        print(f"  α={a:>9.4g}  R²={sc.mean():.4f}±{sc.std():.4f}")
    return pd.DataFrame(rows)


def lasso_path(X, y, *, seed) -> tuple[pd.DataFrame, list, list]:
    print("\n== Lasso α-path (5-fold CV) + coefficient path ==")
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    rows, coef_traces = [], []
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=seed)
    for a in ALPHA_GRID:
        sc = cross_val_score(build_lasso(a, POLY_DEGREE), X, y,
                             cv=cv, scoring="r2", n_jobs=1)
        # fit a single Lasso on Xtr to capture the coefficient vector
        pipe = build_lasso(a, POLY_DEGREE).fit(Xtr, ytr)
        coef = pipe.named_steps["lasso"].coef_
        n_zero = int(np.sum(np.abs(coef) < 1e-8))
        rows.append({"alpha": float(a),
                     "cv_r2_mean": float(sc.mean()),
                     "cv_r2_std": float(sc.std()),
                     "n_nonzero": int(len(coef) - n_zero),
                     "n_zero": n_zero})
        coef_traces.append(np.abs(coef))
        print(f"  α={a:>9.4g}  R²={sc.mean():.4f}±{sc.std():.4f}  "
              f"n_nonzero={len(coef) - n_zero}/{len(coef)}")
    return pd.DataFrame(rows), list(ALPHA_GRID), coef_traces


def plot_lasso_coeff_path(alphas, coef_traces, out_path: Path) -> None:
    matrix = np.stack(coef_traces)
    fig, ax = plt.subplots(figsize=(7, 5))
    for j in range(matrix.shape[1]):
        ax.plot(alphas, matrix[:, j], alpha=0.6, linewidth=1)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel("Lasso α (log)")
    ax.set_ylabel("|coefficient| (symlog)")
    ax.set_title("Lasso coefficient path")
    ax.grid(alpha=0.3)
    save_plot(fig, out_path)


def enet_grid(X, y, *, seed) -> pd.DataFrame:
    print("\n== ElasticNet (α × l1_ratio) grid (5-fold CV) ==")
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for l1 in L1_RATIOS:
        for a in ALPHA_GRID:
            sc = cross_val_score(build_enet(a, l1, POLY_DEGREE), X, y,
                                 cv=cv, scoring="r2", n_jobs=1)
            rows.append({"l1_ratio": float(l1), "alpha": float(a),
                         "cv_r2_mean": float(sc.mean()),
                         "cv_r2_std": float(sc.std())})
        print(f"  l1_ratio={l1:.1f}  done")
    return pd.DataFrame(rows)


def heldout_comparison(X, y, ridge_df, lasso_df, enet_df, *, seed) -> pd.DataFrame:
    print("\n== held-out R² at CV-best α (one split) ==")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    rows = []

    pipe = build_linear(POLY_DEGREE).fit(Xtr, ytr)
    rows.append({"model": "OLS", "best_alpha": float("nan"),
                 "best_l1_ratio": float("nan"),
                 "heldout_r2": float(r2_score(yte, pipe.predict(Xte)))})

    best_r = ridge_df.loc[ridge_df["cv_r2_mean"].idxmax()]
    pipe = build_ridge(float(best_r["alpha"]), POLY_DEGREE).fit(Xtr, ytr)
    rows.append({"model": "Ridge", "best_alpha": float(best_r["alpha"]),
                 "best_l1_ratio": float("nan"),
                 "heldout_r2": float(r2_score(yte, pipe.predict(Xte)))})

    best_l = lasso_df.loc[lasso_df["cv_r2_mean"].idxmax()]
    pipe = build_lasso(float(best_l["alpha"]), POLY_DEGREE).fit(Xtr, ytr)
    rows.append({"model": "Lasso", "best_alpha": float(best_l["alpha"]),
                 "best_l1_ratio": float("nan"),
                 "heldout_r2": float(r2_score(yte, pipe.predict(Xte)))})

    best_e = enet_df.loc[enet_df["cv_r2_mean"].idxmax()]
    pipe = build_enet(float(best_e["alpha"]),
                      float(best_e["l1_ratio"]), POLY_DEGREE).fit(Xtr, ytr)
    rows.append({"model": "ElasticNet", "best_alpha": float(best_e["alpha"]),
                 "best_l1_ratio": float(best_e["l1_ratio"]),
                 "heldout_r2": float(r2_score(yte, pipe.predict(Xte)))})

    for r in rows:
        print(f"  {r['model']:>10s}  α={r['best_alpha']!s:<10s}  "
              f"l1={r['best_l1_ratio']!s:<6s}  R²={r['heldout_r2']:.4f}")
    return pd.DataFrame(rows)


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
    print(f"[data] X={X.shape}  poly degree={POLY_DEGREE}")

    ridge_df = ridge_path(X, y, seed=args.seed)
    save_csv(ridge_df.to_dict(orient="records"), out_dir / "ridge_path.csv")

    lasso_df, alphas, traces = lasso_path(X, y, seed=args.seed)
    save_csv(lasso_df.to_dict(orient="records"), out_dir / "lasso_path.csv")
    plot_lasso_coeff_path(alphas, traces, out_dir / "lasso_coeff_path.png")

    enet_df = enet_grid(X, y, seed=args.seed)
    save_csv(enet_df.to_dict(orient="records"), out_dir / "enet_grid.csv")

    cmp_df = heldout_comparison(X, y, ridge_df, lasso_df, enet_df, seed=args.seed)
    save_csv(cmp_df.to_dict(orient="records"), out_dir / "model_comparison.csv")


if __name__ == "__main__":
    main()
