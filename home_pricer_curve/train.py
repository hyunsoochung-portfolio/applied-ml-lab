"""Linear / polynomial regression R&D study on California Housing.

Experiments:
1. Polynomial degree 1..6 with `cross_val_score` for R² and MAE
   (5-fold). Interaction-only vs full polynomial comparison at each
   degree.
2. Design-matrix condition number per degree as a numerical-stability
   diagnostic.
3. 1-D synthetic extrapolation: kNN flatlines past the training range,
   LinearRegression keeps the slope.

Outputs: ``artifacts/poly_degree_sweep.csv``,
``artifacts/poly_degree_curve.png``,
``artifacts/condition_number.csv``,
``artifacts/extrapolation_demo.png``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import save_csv, save_plot, set_seed  # noqa: E402

from data import load_clean  # noqa: E402
from model import build_linear, build_poly  # noqa: E402


HERE = Path(__file__).resolve().parent
DEGREES = list(range(1, 7))


def _poly_pipe(degree: int, interaction_only: bool) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(
            degree=degree, interaction_only=interaction_only,
            include_bias=False,
        )),
        ("lin", LinearRegression()),
    ])


def degree_sweep(X, y, *, seed: int) -> pd.DataFrame:
    print("\n== polynomial degree sweep (5-fold CV) ==")
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for d in DEGREES:
        for io in (False, True):
            pipe = _poly_pipe(d, io)
            try:
                res = cross_validate(
                    pipe, X, y, cv=cv,
                    scoring=("r2", "neg_mean_absolute_error"),
                    n_jobs=1,
                )
                r2_m = float(res["test_r2"].mean()); r2_s = float(res["test_r2"].std())
                mae_m = float(-res["test_neg_mean_absolute_error"].mean())
                mae_s = float(res["test_neg_mean_absolute_error"].std())
            except Exception as e:
                print(f"  degree={d:>2d} interaction_only={io}: failed ({e!r})")
                r2_m = r2_s = mae_m = mae_s = float("nan")
            rows.append({
                "degree": d, "interaction_only": bool(io),
                "cv_r2_mean": r2_m, "cv_r2_std": r2_s,
                "cv_mae_mean": mae_m, "cv_mae_std": mae_s,
            })
            print(f"  degree={d:>2d}  interaction_only={str(io):>5s}  "
                  f"R²={r2_m:.4f}±{r2_s:.4f}  MAE={mae_m:.4f}")
    return pd.DataFrame(rows)


def plot_degree_curve(df: pd.DataFrame, out_path: Path) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
    for io, marker in ((False, "o"), (True, "s")):
        sub = df[df["interaction_only"] == io].sort_values("degree")
        axL.errorbar(sub["degree"], sub["cv_r2_mean"], yerr=sub["cv_r2_std"],
                     marker=marker, label=f"interaction_only={io}")
        axR.errorbar(sub["degree"], sub["cv_mae_mean"], yerr=sub["cv_mae_std"],
                     marker=marker, label=f"interaction_only={io}")
    axL.set_xlabel("degree"); axL.set_ylabel("CV R²")
    axL.set_title("Polynomial degree vs CV R²"); axL.grid(alpha=0.3); axL.legend()
    axR.set_xlabel("degree"); axR.set_ylabel("CV MAE")
    axR.set_title("Polynomial degree vs CV MAE"); axR.grid(alpha=0.3); axR.legend()
    save_plot(fig, out_path)


def condition_number(X) -> pd.DataFrame:
    print("\n== design-matrix condition number per degree ==")
    rows = []
    # subsample for speed; condition number is a property of column
    # collinearity and is well-approximated on a moderate row count
    Xs = X.sample(n=min(4000, len(X)), random_state=0).to_numpy()
    for d in DEGREES:
        for io in (False, True):
            pf = PolynomialFeatures(degree=d, interaction_only=io,
                                    include_bias=False)
            Z = pf.fit_transform(StandardScaler().fit_transform(Xs))
            try:
                s = np.linalg.svd(Z, compute_uv=False)
                cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
            except Exception:
                cond = float("nan")
            rows.append({
                "degree": d, "interaction_only": bool(io),
                "n_features": int(Z.shape[1]),
                "condition_number": cond,
            })
            print(f"  degree={d:>2d}  interaction_only={str(io):>5s}  "
                  f"n_feat={Z.shape[1]:>4d}  cond={cond:.3e}")
    return pd.DataFrame(rows)


def extrapolation_demo(out_path: Path) -> None:
    print("\n== 1-D extrapolation: kNN vs LinearRegression ==")
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 60)
    y = 1.7 * x + 2.0 + rng.normal(0, 1.5, size=x.size)
    Xtr = x.reshape(-1, 1)
    knn = KNeighborsRegressor(n_neighbors=5).fit(Xtr, y)
    lin = LinearRegression().fit(Xtr, y)
    xq = np.linspace(0, 20, 400).reshape(-1, 1)
    yk = knn.predict(xq); yl = lin.predict(xq)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, color="black", s=24, label="training data")
    ax.axvspan(10, 20, color="orange", alpha=0.08, label="extrapolation zone")
    ax.plot(xq, yk, label="kNN (k=5)")
    ax.plot(xq, yl, label="LinearRegression")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend()
    ax.set_title("kNN cannot extrapolate beyond training range")
    save_plot(fig, out_path)
    print(f"  at x=18: kNN={knn.predict([[18]])[0]:.2f}  "
          f"LinReg={lin.predict([[18]])[0]:.2f}")


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
    print(f"[data] X={X.shape}")

    print("\n== LinearRegression baseline (held-out) ==")
    pipe = build_linear().fit(Xtr, ytr)
    print(f"  train R²={r2_score(ytr, pipe.predict(Xtr)):.4f}  "
          f"test R²={r2_score(yte, pipe.predict(Xte)):.4f}  "
          f"test MAE={mean_absolute_error(yte, pipe.predict(Xte)):.4f}")

    df_sweep = degree_sweep(X, y, seed=args.seed)
    save_csv(df_sweep.to_dict(orient="records"), out_dir / "poly_degree_sweep.csv")
    plot_degree_curve(df_sweep, out_dir / "poly_degree_curve.png")

    df_cond = condition_number(X)
    save_csv(df_cond.to_dict(orient="records"), out_dir / "condition_number.csv")

    extrapolation_demo(out_dir / "extrapolation_demo.png")


if __name__ == "__main__":
    main()
