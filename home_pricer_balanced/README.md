# Home Pricer (Balanced) — Ridge / Lasso / ElasticNet on California Housing

A regularised-regression study on a degree-2 polynomial design: Ridge,
Lasso, ElasticNet — α-paths, coefficient paths, and held-out
comparison at each model's CV-selected α.

## Setup
- `sklearn.datasets.fetch_california_housing` (20 640 rows × 8 features).
- Pipeline: `PolynomialFeatures(degree=2) -> StandardScaler -> <linear model>`.
  The polynomial expansion is fit per CV fold so train-fold rows never
  leak into the design that the validation rows are scored on.
- α grid: `np.logspace(-3, 3, 13)` shared across Ridge / Lasso /
  ElasticNet for a fair comparison.

## Experiments

### 1. Ridge α-path (5-fold CV)
13 α values, `cross_val_score(scoring="r2")` per α.
Output: `ridge_path.csv`.

### 2. Lasso α-path + coefficient path
Same α grid. We additionally fit a single Lasso on an 80/20 split per α
to capture the coefficient vector and plot `|coef|` vs α on a log-symlog
scale.

Outputs: `lasso_path.csv`, `lasso_coeff_path.png`.

### 3. ElasticNet (l1_ratio × α) grid
5 `l1_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` × 13 α = 65 cells.

Output: `enet_grid.csv`.

### 4. Held-out comparison at CV-best α
For each of OLS / Ridge / Lasso / ElasticNet, pick the α (and
`l1_ratio` for ENet) that maximises CV R², refit on the 80% train
split, and report R² on the held-out 20%.

Output: `model_comparison.csv`.

## Findings

This project picks up exactly where `home_pricer_curve` left off: degree-2 polynomial features on California Housing (44 expanded features after standardisation), where OLS catastrophically overfits.

**Ridge α-path (5-fold CV R²):**

| α | R² (mean ± std) |
|---|---|
| 0.001 | -0.242 ± 0.812 |
| 0.01  |  0.128 ± 0.519 |
| 0.10  |  0.574 ± 0.135 |
| **0.316** | **0.6062 ± 0.0704** |
| 1.0   |  0.381 ± 0.281 |
| 10    | -0.645 ± 2.250 |
| 1000  |  0.445 ± 0.313 |

Sweet-spot clearly at α≈0.316. Below it, regularization is too weak and the model picks up the overfit collapse from `home_pricer_curve`; above it, the coefficients are shrunk to a near-mean predictor.

**Lasso α-path** (and the coefficient-sparsity it produces):

| α | R² (mean ± std) | n_nonzero / 44 |
|---|---|---|
| 0.001  | -1.04 ± 3.08 | 28 |
| 0.01   |  0.518 ± 0.168 | 16 |
| **0.0316** | **0.5766 ± 0.0102** | **8** |
| 0.1    |  0.502 ± 0.011 | 3 |
| 0.316  |  0.416 ± 0.008 | 2 |
| 1.0    |  ~0  | 0 |

Lasso's selection arc is sharp: the best CV R² coincides with **8 of 44** features kept active. Past α=1.0 every coefficient is zeroed and the model degenerates to the mean. This is "automatic feature selection" in the strict sense — most of the polynomial expansion was noise, and Lasso's `L1` penalty surfaces the useful eight.

**Held-out R² at CV-best α** (single 80/20 split, same polynomial expansion):

| model | best α | best l1_ratio | held-out R² |
|---|---|---|---|
| OLS         | —      | —    | **−1.1196** |
| Ridge       | 0.316  | —    | **0.6256** |
| Lasso       | 0.0316 | —    | 0.5600 |
| ElasticNet  | 0.0316 | 0.5  | 0.5776 |

OLS collapses by ~1.7 R² compared to its module-05 baseline simply because of the 44-feature expansion — exactly the failure mode regularization fixes. Ridge recovers cleanly (R²=0.6256, slightly better than the linear baseline of 0.5943) by shrinking but keeping all features; Lasso trades a few points of R² for an interpretable sparse model with 8 features; ElasticNet sits between them.

**Design choice:** **Ridge for this problem.** California Housing's features are correlated and informative — there isn't an obvious "kill these eight" structure that justifies Lasso's loss. If interpretability were the priority I'd reach for Lasso's 8-feature model and accept the ~7 pp R² hit.

## Reflections

The single image I'd put on a slide: OLS at R²=−1.12 next to Ridge at R²=0.63 on the *exact same features*. The takeaway isn't "use Ridge" — it's "the same model + a single regularization hyperparameter is the difference between a system that fails catastrophically and one that ships." It's also a clean way to explain why ML projects sometimes need a second iteration that *looks like rework but isn't*.

Lasso's coefficient path also reframes a conversation that often gets stuck. When someone asks "which of the 44 features matter?" the honest answer is "as you tighten the L1 penalty, the model drops them in this order, and at the CV-best α it keeps 8." That's a much more informative answer than a single feature-importance ranking — it explicitly trades off how aggressively you want to simplify. Cross-functionally, that ordering becomes the input to "what data do we actually need to keep collecting?" — a question with real cost implications.

The ElasticNet result (α=0.0316, l1_ratio=0.5 → R²=0.578) is also a small reminder that the answer to "Ridge or Lasso?" is often "tune both and let the data pick." Engineering-wise that's a 5×13 grid search, fast on this dataset; the principle (don't commit to a regularization family before seeing the data) scales.

## Methodology Notes
- All four models share the same `StandardScaler` placement (after
  `PolynomialFeatures`), so the α scales are directly comparable.
- `cross_val_score` returns the **mean of fold scores** — we keep
  `cv_r2_std` alongside so the reader can tell apart "the model is
  better" from "the run is luckier".
- ElasticNet's `l1_ratio=1` reduces to Lasso and `l1_ratio=0` to Ridge;
  we keep the interior of the interval to avoid duplicating the
  endpoint experiments.

## Limitations
- 5-fold single-seed CV — `cv_r2_std` is reported but the α selection
  is from a single seed; for production tuning add `--n-seeds 5` and
  aggregate manually.
- Lasso coefficient path is taken on **one** 80/20 split so the
  coefficient curves are stable across the α grid; the CV scores
  alongside still use the full dataset.

## Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0
python evaluate.py
```
