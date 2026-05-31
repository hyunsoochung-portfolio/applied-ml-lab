# Home Pricer (Neighbor) — kNN Regression on California Housing

Predict median house value with `KNeighborsRegressor`. A 4-dimensional
sweep that goes beyond the basic k curve to look at weights, distance
metric, and scaling jointly.

## Setup
- `sklearn.datasets.fetch_california_housing` (20 640 rows, 8 features,
  target = `MedHouseVal`).
- Pipeline: optional `StandardScaler -> KNeighborsRegressor(...)` built
  by `build_knn()` (`scaled=` flag toggles scaling).

## Experiments

### 1. 4-D grid
`k ∈ {1, 3, 5, 10, 20, 50, 100, 200}` × `weights ∈ {uniform, distance}`
× `metric ∈ {euclidean, manhattan}` × `scaled ∈ {True, False}`.
Logs `train_r2`, `test_r2`, `test_mae` per cell.

Output: `knn_reg_grid.csv` (128 rows).

### 2. Train vs test R² curve over k
Fixed `weights=distance, metric=euclidean, scaled=True` (best non-k
config), sweep `k` and plot the two curves.

Output: `r2_vs_k.png` — the classic high-variance-at-low-k /
high-bias-at-high-k picture.

### 3. Residual plots for best and worst k
The best (highest test R²) and worst (lowest test R²) k values from the
curve above each get a residuals-vs-prediction scatter + residual
histogram on the same held-out test split.

Outputs: `residuals_best.png`, `residuals_worst.png`.

## Findings

**4-D grid winner (California Housing, 20% held-out):**

| k | weights | metric | scaled | train R² | test R² | test MAE |
|---|---|---|---|---|---|---|
| 10 | distance | manhattan | ✓ | 1.000 | **0.7397** | 0.3938 |
| 10 | uniform  | manhattan | ✓ | 0.792 | 0.7366 | 0.3976 |
| 20 | distance | manhattan | ✓ | 1.000 | 0.7324 | 0.4027 |
| 5  | distance | manhattan | ✓ | 1.000 | 0.7290 | 0.4017 |
| 10 | distance | euclidean | ✓ | 1.000 | 0.7041 | 0.4232 |

Two patterns jump out:
- **Manhattan beats euclidean** on California Housing by ~3 pp test R² at the same k. The features (latitude, longitude, median income, room counts) are heterogeneous in scale and meaning even after standardisation — L1 distance is less dominated by any single dimension.
- **`weights="distance"` memorises** (train R² = 1.000 always), but at k=10 the test R² is essentially identical to `weights="uniform"` (0.7397 vs 0.7366). The extra fit on training set buys nothing on test.

**Bias–variance curve** (best non-k config: distance / euclidean / scaled):

| k | train R² | test R² | test MAE |
|---|---|---|---|
| 1   | 1.000 | 0.5276 | 0.5173 |
| 3   | 1.000 | 0.6712 | 0.4411 |
| 5   | 1.000 | 0.6927 | 0.4279 |
| 10  | 1.000 | **0.7041** | 0.4232 |
| 20  | 1.000 | 0.7021 | 0.4291 |
| 50  | 1.000 | 0.6836 | 0.4486 |
| 100 | 1.000 | 0.6634 | 0.4682 |
| 200 | 1.000 | 0.6408 | 0.4917 |

Classic U-curve: k=1 overfits (test R²=0.53, MAE=0.52), k=10 sits on the optimum, k≥50 starts underfitting as neighbourhoods homogenise across heterogeneous housing markets.

**Design choice:** ship `k=10, weights="distance", metric="manhattan", scaled=True` — pareto-best on test R² with no obvious second-place competitor.

## Reflections

The bias-variance curve over `k` is one of the most teachable artifacts in classical ML — train R² flat at 1.0 while test R² traces a U-shape is the textbook picture of overfitting → fit → underfitting on a single dial. I'd open with this plot whenever someone asks "why don't we just memorize all our historical data?" — the answer becomes obvious in 30 seconds, no Bayes-decision-theory required.

Manhattan beating Euclidean on this dataset surfaces a deeper point: distance metrics aren't decorative. California Housing features are heterogeneous in scale and meaning (median income vs latitude vs room counts) even after standardisation; L1 distance is less dominated by any single dimension, so the neighbourhoods it forms cluster more meaningfully. The general rule is that whenever a metric choice moves a test number by 3 pp, there's a *story* about your data underneath — and the story is what most readers actually want, not the number.

On the engineering side, kNN's "train R² = 1.0" with distance weighting looks great in a training log and is meaningless — model memorises the training set by construction. The lesson is that without held-out evaluation, training metrics will *systematically* mislead you toward the most memorising model. This is the same reason I'd never ship a system without integration tests against production-shaped fixtures.

## Methodology Notes
- Single train/test split for the grid (20% test, `random_state=seed`).
  Multi-seed averaging would be cleaner but ~128 cells × 5 seeds × kNN
  on 16k rows starts to bite on a laptop — the bias-variance shape is
  already very stable on a single split given the dataset size.
- "Best" k is selected by `test_r2` because we're presenting the
  bias-variance picture; do not use this k for downstream production
  selection (that's what CV is for; see `cardio_protocol`).

## Limitations
- California Housing's residuals are heteroscedastic and the target is
  clipped at 5.0 — both visible in the best-k residual plot. Real
  diagnostics would use QQ-plots and weighted error metrics; kept
  simple here.
- `k=1` is a sanity check, not a serious model.

## Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0
python evaluate.py
```
