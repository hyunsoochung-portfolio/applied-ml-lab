# Wholesale Segmenter — Feature Scaling on UCI Wholesale Customers

A scaling-matters study that goes wider than a single algorithm: we
build the full scaler × algorithm matrix, then probe robustness with an
outlier-injection ablation.

## Setup
- UCI Wholesale Customers
- URL: <https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv>
- Target: `Channel` (1=Horeca/0, 2=Retail/1), 440 rows, 6 numeric features
- Raw column std spans roughly two orders of magnitude — a clean
  textbook case for distance-based learners to break without scaling.

## Experiments

### 1. Scaler × algorithm matrix
- Scalers: `{none, standard, minmax, robust, maxabs}`
- Algorithms: `{kNN, LinearSVC, LogReg, RandomForest}`
- 5 seeds × `StratifiedKFold(5)` per cell → 25 fold scores per cell
- Reports `acc_mean ± acc_std` and total fit / predict wall-clock

Outputs:
- `scaler_algo_matrix.csv` — one row per (algorithm, scaler) cell
- `scaler_algo_heatmap.png` — visual summary
- `timings.csv` — wall-clock per cell

### 2. Outlier injection ablation
For each seed and each outlier fraction `f ∈ {0, 5%, 10%, 20%}`, we
inject `f * n_train` rows whose values on a random subset of columns are
multiplied by 50× (extreme but not unrealistic for an inflated-receipt
data-entry scenario). We then re-evaluate StandardScaler vs RobustScaler
across all four algorithms on a clean held-out test set.

Outputs:
- `outlier_ablation.csv` — `(seed, outlier_frac, scaler, algorithm, test_acc)`
- `outlier_ablation.png` — 2×2 panel, one subplot per algorithm

### 3. Headline kNN flip demo
Carried over from the textbook example: show that at least one test
prediction changes when the scaler changes.

## Findings

**Scaler × algorithm matrix** (3-seed StratifiedKFold mean accuracy on Wholesale Customers):

| algorithm \ scaler | none | standard | minmax | robust | maxabs |
|---|---|---|---|---|---|
| **kNN**           | 0.8848 | 0.9136 | 0.9129 | 0.9098 | 0.9129 |
| **LinearSVC**     | 0.8894 | 0.9045 | 0.8864 | 0.9061 | 0.8864 |
| **LogReg**        | 0.9076 | 0.9023 | 0.8341 | 0.9038 | 0.8341 |
| **RandomForest**  | 0.9121 | 0.9121 | 0.9121 | 0.9121 | 0.9121 |

Reading the matrix:
- **RandomForest is scale-invariant**, as theory predicts — every cell is bit-for-bit identical (0.9121). Trees split on thresholds, not distances.
- **kNN gains ~3 pp from any scaling** (0.8848 → 0.9136 with StandardScaler). Without scaling the highest-variance feature dominates the distance metric and the model becomes effectively 1-D.
- **LinearSVC** prefers `robust` (0.9061) — Wholesale Customers has a few large purchase outliers, and the IQR-based scaler stops them dominating the margin.
- **LogReg** is unintuitive here: `none` ties for best (0.9076) and `minmax`/`maxabs` collapse to 0.8341. Min/max anchoring on outlier-heavy features actively distorts the ranges the linear model sees.

**Outlier injection ablation** (5%/10%/20% extreme-row injection, kept best scaler per algorithm):

| algorithm | clean | 5% outliers | 10% | 20% |
|---|---|---|---|---|
| LinearSVC (robust)  | 0.9061 | 0.8182 | 0.6939 | 0.6818 |
| LogReg (robust)     | 0.9091 | 0.8697 | 0.7697 | 0.6848 |
| RandomForest        | 0.9091 | 0.9061 | 0.9000 | 0.9030 |
| kNN                 | 0.8970 | 0.8970 | 0.9030 | 0.8970 |

Linear models collapse under heavy outlier contamination even with `RobustScaler`; trees and kNN stay flat because their decision surfaces aren't pulled by extreme magnitudes. The scaler choice only matters until outlier mass crosses ~10% — past that, the *algorithm family* is what saves you.

**kNN flip demo (k=5):** at row 13 of the held-out split, `none` predicts the wrong class (`1`) while every scaler flips it to `0` — concrete instance of "scaling changes labels, not just probabilities."

## Reflections

The scaler × algorithm matrix is one of the most useful things a junior engineer can run on a new dataset. The conventional wisdom — "always scale" — is wrong in detail (RandomForest is invariant, LogReg can actively regress) and the matrix gives you the *real* rule: it depends on the algorithm's distance / margin / split-rule semantics. Internalising that one matrix prevents whole categories of "I scaled and it got worse, what did I do wrong" tickets.

The outlier ablation tells the story I'd put on the first slide when picking an algorithm: with 5% outliers, linear models lose ~9 pp accuracy; with 20% outliers, they collapse to near-random. Trees and kNN stay flat. So "which model" isn't a pure modelling question — it's a question about your data hygiene pipeline. If we don't know the outlier rate in the live system, the safer choice is the algorithm family that doesn't care.

The kNN flip demo is also a small communication tool I'd reach for first: a single row where the prediction *literally changes* depending on scaler choice. "It's not that the accuracy gets better — the actual label flips on real rows" is more visceral than aggregate metrics for anyone who hasn't internalised "distance-based models without scaling are nearly 1-D."

## Methodology Notes
- Scaler always lives **inside** the `Pipeline`, so cross-validation
  fits scaling on train folds only.
- Outlier injection only mutates the train split — the test split is
  unchanged, so the comparison isolates the scaler's robustness rather
  than rewarding any scheme that happens to match the perturbation.
- `LinearSVC` uses `dual="auto"` (sklearn 1.4+ default chooser) to
  avoid the convergence-warning noise on small `n`.

## Limitations
- 440 rows is small; per-cell std is meaningful, but absolute
  differences in the 1-2 pt range should not be over-interpreted.
- The outlier injection is synthetic — real-world outliers are usually
  more structured (a single bad sensor / a single store). Treat the
  shape of the degradation curves as the takeaway, not the absolute
  drop magnitude.

## Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0 --n-seeds 5
python evaluate.py
```
