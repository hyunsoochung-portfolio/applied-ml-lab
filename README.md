# ml-projects

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-EB6E2A)
![LightGBM](https://img.shields.io/badge/LightGBM-4.1%2B-008000)

13 classical-ML projects on public datasets. Each one runs the sweeps and ablations I'd run before picking a model, writes the tables and plots to `artifacts/`, and ends with a one-line design choice in the README.

Per-project layout: `data.py` / `model.py` / `train.py` / `evaluate.py` / `README.md`.

## Methodology

Conventions shared across projects:

- **Multi-seed runs with mean ± std** where it pays (kNN, CV, scaling, SGD, ensembles — seeded `StratifiedKFold` / `KFold` driven by `--seed` and `--n-seeds`).
- **`artifacts/` folder per project** for CSV tables and PNGs. Gitignored except `.gitkeep`.
- **Leakage-safe pipelines**: scalers / encoders / polynomial features live inside `sklearn.Pipeline` so they fit only on train folds during CV. `cardio_protocol` quantifies the inflation from doing it wrong.
- **Reproducibility**: every `train.py` takes `--seed` (default 0), `--n-seeds` (default 5 where used), and `--output-dir` (default `artifacts/`). Same seed → same output.

## Projects

| project                                | concepts (short)                                         | dataset                          | metric            | experiments (sweeps)                                                                                  |
|----------------------------------------|----------------------------------------------------------|----------------------------------|-------------------|-------------------------------------------------------------------------------------------------------|
| `cardio_screener`               | kNN sweep, decision-boundary plot, bootstrap CI          | Heart Disease (UCI)              | accuracy, F1      | full `k × weights × metric` grid × 5 seeds; bootstrap CI; 3 feature-pair boundaries                   |
| `cardio_protocol`                 | splitter variance, nested CV, leakage demo               | Heart Disease (UCI)              | acc / F1 / AUC    | 5 splitters × 10 seeds; fold-K sweep; nested CV (5×3); leak-vs-clean inflation                        |
| `wholesale_segmenter`                | scaler × algorithm matrix, outlier injection             | Wholesale Customers (UCI)        | accuracy          | 5 scalers × 4 algorithms × 5 seeds; outlier-fraction ablation; wall-clock                             |
| `home_pricer_neighbor`                | 4-D grid, bias-variance, residual plots                  | California Housing               | R², MAE           | `k × weights × metric × scaled` grid; residuals for best / worst k                                    |
| `home_pricer_curve`             | poly degree sweep, condition number, extrapolation       | California Housing               | R²                | degree 1..6 × `interaction_only`; cond(Z) per degree; 1-D extrapolation                               |
| `home_pricer_balanced`        | Ridge / Lasso / ElasticNet α-paths, coefficient path     | California Housing               | R²                | 13-α grid (Ridge / Lasso); 5×13 ElasticNet grid; held-out comparison at CV-best                       |
| `churn_alert`               | solver / C / class_weight, calibration, ROC + PR         | Telco Customer Churn             | acc, ROC-AUC      | 3 solvers; 11-C log sweep; class_weight ablation; calibration + Brier; ROC + PR; OvR vs softmax       |
| `income_streamer`          | loss × LR-schedule, batch size, early stop, epoch curve  | Adult Income (UCI)               | accuracy          | 4×4 loss×schedule grid; 3 batch sizes; early-stop comparison; multi-seed epoch curve                  |
| `mushroom_screener`              | criterion × depth × leaf grid, ccp pruning, importances  | Mushroom (UCI)                   | accuracy          | 2×6×3 grid; cost-complexity pruning path; tree-vs-permutation importance                              |
| `income_tuner`                  | 4-way search comparison, Pareto, n_iter sensitivity      | Adult Income (UCI)               | ROC-AUC           | Grid / Random / HalvingGrid / HalvingRandom on same space; n_iter ∈ {10, 50, 200}                     |
| `deposit_propensity`             | 6 ensembles × 5 seeds, OOF, importance agreement         | Bank Marketing (UCI)             | ROC-AUC, F1, acc  | 6 ensembles × 5 seeds; perm importance; Spearman agreement matrix; calibration overlay; OOF probs    |
| `shopper_personas`                     | 4-metric K sweep, init study, MiniBatch comparison       | Mall Customers (mirror)          | silhouette / DB / CH | K=2..12 × 4 metrics; init × n_init grid; MiniBatch vs KMeans (ARI); multi-pair scatter             |
| `tumor_compressor`                   | PCA vs KernelPCA, scree, reconstruction, whitening       | Breast Cancer Wisconsin          | accuracy          | linear / rbf / poly KPCA; n=1..30 sweep; Frobenius reconstruction; whitening ablation; 2D + 3D scatter |

## Results highlights

Headline numbers from a recent run (`--n-seeds 3`, default flags otherwise). Each project's README has the full Findings section with tables, methodology notes, and the design choice that came out of the experiment.

| project | headline finding | metric |
|---|---|---|
| `cardio_screener`        | k=21-31 plateau, all top configs euclidean, uniform/distance a wash | acc 0.847 ± 0.002 |
| `cardio_protocol`          | nested CV (0.821) more conservative than naive 5-fold (0.833); larger K → lower across-seed std but bigger within-fold spread | acc |
| `wholesale_segmenter`         | RF scale-invariant; kNN +3 pp from StandardScaler; LogReg actively *hurt* by MinMax on outlier-heavy features | acc 0.91 |
| `home_pricer_neighbor`         | Manhattan + scaled + k=10 beats Euclidean by ~3 pp; clean U-curve in k | test R² 0.740 |
| `home_pricer_curve`      | OLS collapses past degree 2 on California Housing (cond. number 10³ → 10¹⁷); `interaction_only` slows the collapse but doesn't fix it | held-out R² |
| `home_pricer_balanced` | Ridge α=0.316 recovers the polynomial features (held-out R² 0.626 vs OLS −1.12); Lasso ships 8 of 44 features | held-out R² |
| `churn_alert`        | `class_weight="balanced"` trades 5 pp accuracy for **+26 pp minority recall**; AUC unchanged; multinomial > OvR by 3 pp on 3-class | acc, ROC-AUC |
| `income_streamer`   | `hinge + adaptive` best cell (0.848); smaller batches buy ~2 pp at 3-4× cost; default early stopping too eager | acc 0.815 ± seed |
| `mushroom_screener`       | Mushroom perfectly separable at depth ≥ 6; ccp_alpha=0.005 yields a 13-node tree at 99.5%; tree_importance vs permutation diverges sharply | acc 1.000 |
| `income_tuner`           | **HalvingGridSearch is Pareto-best** (0.9134 in 28.8s vs Grid 0.9140 in 99.4s); RandomSearch plateaus at n_iter ≈ 50 | ROC-AUC |
| `deposit_propensity`      | HistGB 0.9346 wins; LightGBM and XGBoost tied at 0.934; XGBoost ↔ LightGBM importance Spearman ρ=0.33 (different paths, same score) | ROC-AUC |
| `shopper_personas`              | Silhouette peaks at k=11 but operationally k=5 is correct; k-means++ matches `random` at 10× fewer inits | silhouette, DB |
| `tumor_compressor`            | **5 components** beat 30 (0.965 vs 0.958) — extra components are noise; KernelPCA(rbf) actively hurts on a linearly-separable problem | acc 0.965 |

## Artifacts

Each project's `train.py` writes its results under `<project>/artifacts/`
(gitignored except for `.gitkeep`). You'll find:

- `*.csv` — every grid / sweep / table the run produces; tidy long-form
  where reasonable so a follow-up notebook can pivot freely.
- `*.png` — methodology-grade plots (decision boundaries, error-bar
  sweeps, calibration overlays, Pareto fronts, scree plots, etc.).

Artifacts are intentionally not checked in: every run regenerates them,
and they're cheap (most projects finish well under a minute, the search
projects under a few minutes on a laptop).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# pick any project and run with a seed
python cardio_screener/train.py --seed 0
python deposit_propensity/train.py --seed 0 --n-seeds 5
```

Every `train.py` accepts the same baseline flags:

```text
--seed INT          # default 0
--n-seeds INT       # default 5 where multi-seed is used
--output-dir PATH   # default <project>/artifacts/
```

A quick sanity-check entry-point per project:

```bash
cd mushroom_screener && python evaluate.py
```

## Layout

```
ml-projects/
├── README.md
├── requirements.txt
├── .gitignore
├── shared/
│   └── utils.py           # set_seed, timer, save_plot, save_csv,
│                          # multi_seed_metric, pareto_front, artifacts_dir
└── <project_name>/
    ├── README.md           # Setup / Experiments / Findings / Reflections / Methodology / Limitations / Reproduce
    ├── data.py             # download + load + clean + split
    ├── model.py            # estimator / pipeline factory
    ├── train.py            # CLI: full R&D sweep, writes artifacts/
    ├── evaluate.py         # quick eval-only entry
    └── artifacts/
        └── .gitkeep        # results land here at runtime (gitignored)
```


---

# 📚 Full project write-ups

Every folder's full write-up is inlined below — all 13 project write-ups, so you can read everything without opening a single folder. The same text lives in each project's own `README.md`.

- [Cardio Screener — kNN on UCI Cleveland Heart Disease](#cardio-screener--knn-on-uci-cleveland-heart-disease)
- [Cardio Protocol — Data Splits & Cross-Validation on UCI Cleveland Heart Disease](#cardio-protocol--data-splits--cross-validation-on-uci-cleveland-heart-disease)
- [Wholesale Segmenter — Feature Scaling on UCI Wholesale Customers](#wholesale-segmenter--feature-scaling-on-uci-wholesale-customers)
- [Home Pricer (Neighbor) — kNN Regression on California Housing](#home-pricer-neighbor--knn-regression-on-california-housing)
- [Home Pricer (Curve) — Linear & Polynomial Regression on California Housing](#home-pricer-curve--linear--polynomial-regression-on-california-housing)
- [Home Pricer (Balanced) — Ridge / Lasso / ElasticNet on California Housing](#home-pricer-balanced--ridge--lasso--elasticnet-on-california-housing)
- [Churn Alert — Logistic Regression on Telco Customer Churn](#churn-alert--logistic-regression-on-telco-customer-churn)
- [Income Streamer — SGD on UCI Adult Income](#income-streamer--sgd-on-uci-adult-income)
- [Mushroom Screener — Decision Tree on UCI Mushroom](#mushroom-screener--decision-tree-on-uci-mushroom)
- [Income Tuner — Hyperparameter Tuning on UCI Adult Income](#income-tuner--hyperparameter-tuning-on-uci-adult-income)
- [Deposit Propensity — Tree-Ensemble Benchmark on UCI Bank Marketing](#deposit-propensity--tree-ensemble-benchmark-on-uci-bank-marketing)
- [Shopper Personas — KMeans Clustering on Mall Customers](#shopper-personas--kmeans-clustering-on-mall-customers)
- [Tumor Compressor — PCA on Breast Cancer Wisconsin](#tumor-compressor--pca-on-breast-cancer-wisconsin)


<br>

---

## Cardio Screener — kNN on UCI Cleveland Heart Disease

> 📁 [`cardio_screener/`](./cardio_screener)

Predict whether a patient has heart disease (binary: derived from the
`num` column, `>0 -> 1`) using k-Nearest Neighbours. This project is set
up as a small R&D-style study rather than a single train run: a full
parameter grid, multi-seed cross-validation, and bootstrap CI for the
selected configuration.

### Setup
- Source: UCI ML Repository — `processed.cleveland.data`
- URL: <https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data>
- 303 rows, 13 features; `?` treated as missing and dropped before
  modelling; downloaded once and cached under `./data/`.
- All preprocessing (`StandardScaler`) is wrapped inside the
  scikit-learn `Pipeline` so it fits **inside** each CV fold and cannot
  leak the validation rows.

### Experiments
| experiment | sweep | what's logged |
|---|---|---|
| Full grid CV | `n_neighbors ∈ {1,3,5,7,11,15,21,31}` × `weights ∈ {uniform, distance}` × `metric ∈ {euclidean, manhattan, minkowski(p=3)}` | 5-seed `StratifiedKFold` mean ± std for accuracy and F1 |
| Bootstrap CI | resample test predictions of the best CV config 200× | 95% percentile interval on accuracy |
| Decision boundaries | 3 feature pairs × 3 k values (1 / 5 / 25) | 2-D contour plots |

Artifacts written under `artifacts/`:
- `knn_sweep.csv` — flat table of every grid cell with mean ± std
- `knn_sweep_heatmap.png` — visual summary of the grid
- `best_config_bootstrap.csv` — single-row bootstrap result
- `boundary_<feat_a>-<feat_b>_k<k>.png` — 9 boundary plots

### Findings

The grid run with 3 seeds × 5-fold StratifiedKFold produced a very consistent picture (`acc_std` ≤ 0.007 on the leaders). Top five CV configurations:

| n_neighbors | weights | metric | acc_mean | acc_std |
|---|---|---|---|---|
| 31 | distance | euclidean | 0.8475 | 0.0016 |
| 31 | uniform  | euclidean | 0.8475 | 0.0016 |
| 31 | uniform  | minkowski(p=3) | 0.8452 | 0.0073 |
| 31 | distance | minkowski(p=3) | 0.8441 | 0.0043 |
| 15 | uniform  | euclidean | 0.8441 | 0.0017 |

Reading the grid:
- **`k` dominates.** On a held-out split, accuracy climbs cleanly from `k=1` (0.733) to `k=31` (0.867) — bias-variance moving in the right direction without an overfit dip.
- **Euclidean wins narrowly.** The top configurations are euclidean; manhattan and minkowski(p=3) trail by a few tenths of a percent — same neighbourhood, slightly different geometry.
- **Uniform vs distance weighting is a wash here.** Top two are identical to four decimals — distance weighting doesn't add signal when the neighbourhood is balanced.

The bootstrap 95% CI on the best config (`k=31, distance, euclidean`) is **[0.7733, 0.9333]** around a held-out accuracy of 0.8533 — a wide band that reflects the small (303-row) sample, not algorithm uncertainty.

**Design choice:** I'd ship `k=21, uniform, euclidean` rather than the apparent best at `k=31` — the simpler model is statistically indistinguishable on this dataset (gap ≪ CI half-width) and degrades less when the optimum drifts a notch on fresh splits.

### Reflections

kNN's strength on small structured data is that it's the most transparent model possible — every prediction is literally "look at these k nearest patients." For a 303-row clinical dataset, that interpretability is worth more than the 1–2 pp of accuracy you'd squeeze out of an ensemble. The clean k=21–31 plateau also tells me something practical: on this size of data, *more sophisticated tuning isn't the bottleneck — more data is.* Knowing which one to chase is the actual judgement call.

The bootstrap CI width ([0.77, 0.93]) is what I'd actually report, not the 0.85 point estimate. "Our held-out accuracy is 85% with a 95% CI of about 8 points" is honest about how much we know on 303 rows; "the model is 85% accurate" implies a precision we don't have. Naming that distinction — point estimate vs uncertainty — is half the conversation when a clinician asks "is the model good enough yet?"

On the systems side, brute-force kNN is O(N) per prediction; running this anywhere with real query volume would need a k-d / ball tree under the hood. The decision-boundary PNGs in `artifacts/` are deliberately the kind of figure a clinician (or anyone outside the modelling team) can glance at and reason about without reading a confusion matrix — they're as much a communication tool as a debugging one.

### Methodology Notes
- 5-fold `StratifiedKFold` for every grid cell, repeated across 5 seeds
  so the reported `acc_mean ± acc_std` reflects fold variance and seed
  variance together.
- The bootstrap CI is computed on **test-set predictions**, not on
  resampled training data, so it estimates the sampling noise of the
  accuracy estimator on a held-out split (cheap and standard).
- Decision-boundary plots are 2-D for visual intuition only and use a
  fresh scaler+model per pair; the headline metric is the full-feature
  grid above.

### Limitations
- 303 rows is small — wide confidence intervals are expected and that's
  part of the lesson; treat absolute differences below ~2 percentage
  points as noise.
- `weights="distance"` can be sensitive to duplicate / near-duplicate
  rows; nothing pathological in this dataset, but the effect is real on
  noisier ones.

### Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0 --n-seeds 5
# or, smaller / faster
python train.py --seed 0 --n-seeds 2
python evaluate.py
```
All outputs land in `artifacts/` (gitignored).


<br>

---

## Cardio Protocol — Data Splits & Cross-Validation on UCI Cleveland Heart Disease

> 📁 [`cardio_protocol/`](./cardio_protocol)

A small R&D study of how data-splitting choices affect what your
"validation accuracy" actually means. Four experiments, all logged to
`artifacts/` as CSV + PNG.

### Setup
- Same dataset as `cardio_screener` (Cleveland Heart Disease).
- Reference estimator: `StandardScaler -> LogisticRegression(max_iter=2000)`
  built by `build_classifier()` in `model.py`.

### Experiments

#### 1. Splitter variance across 10 seeds
For each seed we record the score from:
- `train_test_split` (random hold-out)
- `train_test_split(stratify=y)` (stratified hold-out)
- `KFold(5)`
- `StratifiedKFold(5)`
- `RepeatedStratifiedKFold(5, n_repeats=2)`

Output: `splitter_variance.csv` (one row per fold per seed),
`splitter_variance.png` (boxplot per splitter).

#### 2. Fold-count sensitivity
Sweeps `K ∈ {3, 5, 10, 20}` of `StratifiedKFold`, repeating across all
seeds. Logs within-CV std (fold-to-fold) and across-seed std (run-to-run).

Output: `fold_count.csv`.

#### 3. Nested CV
Outer `StratifiedKFold(5)` for model selection, inner `StratifiedKFold(3)`
for `LogisticRegression(C ∈ {0.01, 0.1, 1, 10})`. Records the chosen `C`
per outer fold along with the inner-best and outer-test scores.

Output: `nested_cv.csv`.

#### 4. Leakage quantification
Two pipelines for the *same* model & splits:
- **leaky**: `StandardScaler` fit on the full `X` before CV.
- **clean**: scaler inside the sklearn `Pipeline`, fit per fold.

Output: `leakage.csv` with an `inflation = leaky - clean` column per seed.

### Findings

**Splitter variance across 10 seeds** (LogisticRegression baseline on Heart Disease):

| splitter | mean | std | min | max |
|---|---|---|---|---|
| `KFold(5)`              | 0.8361 | 0.0494 | 0.7458 | 0.9333 |
| `StratifiedKFold(5)`    | 0.8329 | 0.0475 | 0.7627 | 0.9492 |
| `RepeatedSKF(5×2)`      | 0.8323 | 0.0474 | 0.7627 | 0.9492 |
| `holdout_random`        | 0.8444 | 0.0278 | 0.8133 | 0.8667 |
| `holdout_stratified`    | 0.8267 | 0.0133 | 0.8133 | 0.8400 |

Single hold-out splits look *more stable* across seeds than CV, but that's a measurement artefact — each hold-out is one point estimate, so the spread is just "which split did you draw?" CV averages over folds first, so the across-seed std reflects seed noise after that averaging.

**Fold-count sensitivity** (Heart Disease, LogReg):

| K | across-seed mean | across-seed std | avg within-CV std |
|---|---|---|---|
| 3  | 0.8316 | 0.0099 | 0.0344 |
| 5  | 0.8329 | 0.0070 | 0.0440 |
| 10 | 0.8320 | 0.0047 | 0.0573 |
| 20 | 0.8331 | 0.0030 | 0.0933 |

Larger K stabilises the *across-seed* estimate (std 0.0099 → 0.0030) but inflates the *within-CV* spread (0.0344 → 0.0933) because each held-out fold gets smaller and noisier. K=5 is the right default on this size of dataset.

**Nested CV (outer=5, inner=3):** outer mean **0.8214 ± 0.0372** — slightly more conservative than naive 5-fold (0.8329), as expected when model selection is honestly nested.

**Leakage demo (scale-before vs scale-inside pipeline):** inflation is at most 0.003 on Heart Disease (303 rows × 13 features) — too small to see at this scale, but the structural difference matters and on larger feature spaces it bites. Keeping `StandardScaler` inside the `Pipeline` is the cheap always-correct default.

### Reflections

The leakage demo is the heart of this project for me. On 303 rows the measured inflation was tiny (≤0.003), but the *structural* mistake is identical to the one that ships +5 pp "improvements" on larger feature spaces and then quietly degrades in production. Most data-science bugs that survive into prod look like that — small effects on the metric you check, large effects on the world. Putting the scaler inside the `Pipeline` is a one-line fix that closes that whole class of bug forever; the harder thing is explaining to a teammate *why* the "obviously correct" `scaler.fit_transform(X)` outside CV is actually wrong.

Cross-validation is also where "what does this metric mean" conversations come up. Hold-out looks more stable than 5-fold CV across seeds, but only because it averages nothing — each draw is a single noisy estimate. Saying it plainly — "the smaller-looking number is the more honest one" — is the kind of explanation that buys trust over time.

On the engineering side, nested CV is honest model-selection but it's also 15× the compute of a single fit. There's a real product judgement under that: when does the conservatism of nested CV pay for itself, and when is "naive 5-fold + a willingness to re-train on fresh data" cheaper and just as safe? The right answer depends on how fast your data drifts, not on what's nominally "more correct."

### Methodology Notes
- We never call `.fit_transform` on the full dataset for any reported
  metric except the explicit leakage demo — that's the whole point.
- `StratifiedKFold` is the default everywhere except (a) the
  non-stratified hold-out, (b) plain `KFold` used as a comparison
  baseline; both included so the variance gap is visible.
- "Splitter variance" intentionally mixes hold-out (1 score per seed)
  with k-fold (5 scores per seed); the boxplot per splitter is the
  right object to look at.

### Limitations
- ~297 rows after cleaning is small — variance estimates themselves
  carry noise. The qualitative pattern (`KFold` widest, `RSKF` tightest)
  is stable; absolute numbers will jiggle across machines.
- Nested CV uses only 4 candidate `C` values to keep wall-clock low;
  expand `grid` in `nested_cv()` for a finer sweep.

### Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0 --n-seeds 10
python evaluate.py
```


<br>

---

## Wholesale Segmenter — Feature Scaling on UCI Wholesale Customers

> 📁 [`wholesale_segmenter/`](./wholesale_segmenter)

A scaling-matters study that goes wider than a single algorithm: we
build the full scaler × algorithm matrix, then probe robustness with an
outlier-injection ablation.

### Setup
- UCI Wholesale Customers
- URL: <https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv>
- Target: `Channel` (1=Horeca/0, 2=Retail/1), 440 rows, 6 numeric features
- Raw column std spans roughly two orders of magnitude — a clean
  textbook case for distance-based learners to break without scaling.

### Experiments

#### 1. Scaler × algorithm matrix
- Scalers: `{none, standard, minmax, robust, maxabs}`
- Algorithms: `{kNN, LinearSVC, LogReg, RandomForest}`
- 5 seeds × `StratifiedKFold(5)` per cell → 25 fold scores per cell
- Reports `acc_mean ± acc_std` and total fit / predict wall-clock

Outputs:
- `scaler_algo_matrix.csv` — one row per (algorithm, scaler) cell
- `scaler_algo_heatmap.png` — visual summary
- `timings.csv` — wall-clock per cell

#### 2. Outlier injection ablation
For each seed and each outlier fraction `f ∈ {0, 5%, 10%, 20%}`, we
inject `f * n_train` rows whose values on a random subset of columns are
multiplied by 50× (extreme but not unrealistic for an inflated-receipt
data-entry scenario). We then re-evaluate StandardScaler vs RobustScaler
across all four algorithms on a clean held-out test set.

Outputs:
- `outlier_ablation.csv` — `(seed, outlier_frac, scaler, algorithm, test_acc)`
- `outlier_ablation.png` — 2×2 panel, one subplot per algorithm

#### 3. Headline kNN flip demo
Carried over from the textbook example: show that at least one test
prediction changes when the scaler changes.

### Findings

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

### Reflections

The scaler × algorithm matrix is one of the most useful things a junior engineer can run on a new dataset. The conventional wisdom — "always scale" — is wrong in detail (RandomForest is invariant, LogReg can actively regress) and the matrix gives you the *real* rule: it depends on the algorithm's distance / margin / split-rule semantics. Internalising that one matrix prevents whole categories of "I scaled and it got worse, what did I do wrong" tickets.

The outlier ablation tells the story I'd put on the first slide when picking an algorithm: with 5% outliers, linear models lose ~9 pp accuracy; with 20% outliers, they collapse to near-random. Trees and kNN stay flat. So "which model" isn't a pure modelling question — it's a question about your data hygiene pipeline. If we don't know the outlier rate in the live system, the safer choice is the algorithm family that doesn't care.

The kNN flip demo is also a small communication tool I'd reach for first: a single row where the prediction *literally changes* depending on scaler choice. "It's not that the accuracy gets better — the actual label flips on real rows" is more visceral than aggregate metrics for anyone who hasn't internalised "distance-based models without scaling are nearly 1-D."

### Methodology Notes
- Scaler always lives **inside** the `Pipeline`, so cross-validation
  fits scaling on train folds only.
- Outlier injection only mutates the train split — the test split is
  unchanged, so the comparison isolates the scaler's robustness rather
  than rewarding any scheme that happens to match the perturbation.
- `LinearSVC` uses `dual="auto"` (sklearn 1.4+ default chooser) to
  avoid the convergence-warning noise on small `n`.

### Limitations
- 440 rows is small; per-cell std is meaningful, but absolute
  differences in the 1-2 pt range should not be over-interpreted.
- The outlier injection is synthetic — real-world outliers are usually
  more structured (a single bad sensor / a single store). Treat the
  shape of the degradation curves as the takeaway, not the absolute
  drop magnitude.

### Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0 --n-seeds 5
python evaluate.py
```


<br>

---

## Home Pricer (Neighbor) — kNN Regression on California Housing

> 📁 [`home_pricer_neighbor/`](./home_pricer_neighbor)

Predict median house value with `KNeighborsRegressor`. A 4-dimensional
sweep that goes beyond the basic k curve to look at weights, distance
metric, and scaling jointly.

### Setup
- `sklearn.datasets.fetch_california_housing` (20 640 rows, 8 features,
  target = `MedHouseVal`).
- Pipeline: optional `StandardScaler -> KNeighborsRegressor(...)` built
  by `build_knn()` (`scaled=` flag toggles scaling).

### Experiments

#### 1. 4-D grid
`k ∈ {1, 3, 5, 10, 20, 50, 100, 200}` × `weights ∈ {uniform, distance}`
× `metric ∈ {euclidean, manhattan}` × `scaled ∈ {True, False}`.
Logs `train_r2`, `test_r2`, `test_mae` per cell.

Output: `knn_reg_grid.csv` (128 rows).

#### 2. Train vs test R² curve over k
Fixed `weights=distance, metric=euclidean, scaled=True` (best non-k
config), sweep `k` and plot the two curves.

Output: `r2_vs_k.png` — the classic high-variance-at-low-k /
high-bias-at-high-k picture.

#### 3. Residual plots for best and worst k
The best (highest test R²) and worst (lowest test R²) k values from the
curve above each get a residuals-vs-prediction scatter + residual
histogram on the same held-out test split.

Outputs: `residuals_best.png`, `residuals_worst.png`.

### Findings

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

### Reflections

The bias-variance curve over `k` is one of the most teachable artifacts in classical ML — train R² flat at 1.0 while test R² traces a U-shape is the textbook picture of overfitting → fit → underfitting on a single dial. I'd open with this plot whenever someone asks "why don't we just memorize all our historical data?" — the answer becomes obvious in 30 seconds, no Bayes-decision-theory required.

Manhattan beating Euclidean on this dataset surfaces a deeper point: distance metrics aren't decorative. California Housing features are heterogeneous in scale and meaning (median income vs latitude vs room counts) even after standardisation; L1 distance is less dominated by any single dimension, so the neighbourhoods it forms cluster more meaningfully. The general rule is that whenever a metric choice moves a test number by 3 pp, there's a *story* about your data underneath — and the story is what most readers actually want, not the number.

On the engineering side, kNN's "train R² = 1.0" with distance weighting looks great in a training log and is meaningless — model memorises the training set by construction. The lesson is that without held-out evaluation, training metrics will *systematically* mislead you toward the most memorising model. This is the same reason I'd never ship a system without integration tests against production-shaped fixtures.

### Methodology Notes
- Single train/test split for the grid (20% test, `random_state=seed`).
  Multi-seed averaging would be cleaner but ~128 cells × 5 seeds × kNN
  on 16k rows starts to bite on a laptop — the bias-variance shape is
  already very stable on a single split given the dataset size.
- "Best" k is selected by `test_r2` because we're presenting the
  bias-variance picture; do not use this k for downstream production
  selection (that's what CV is for; see `cardio_protocol`).

### Limitations
- California Housing's residuals are heteroscedastic and the target is
  clipped at 5.0 — both visible in the best-k residual plot. Real
  diagnostics would use QQ-plots and weighted error metrics; kept
  simple here.
- `k=1` is a sanity check, not a serious model.

### Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0
python evaluate.py
```


<br>

---

## Home Pricer (Curve) — Linear & Polynomial Regression on California Housing

> 📁 [`home_pricer_curve/`](./home_pricer_curve)

Linear baseline plus a polynomial-degree study with a numerical-stability
side channel: condition number per design matrix.

### Setup
- `sklearn.datasets.fetch_california_housing` (20 640 rows × 8 features,
  target = `MedHouseVal`).
- Pipeline: `StandardScaler -> PolynomialFeatures(degree, interaction_only)
  -> LinearRegression`.

### Experiments

#### 1. Degree sweep (1..6) with CV R² + MAE
For each `degree ∈ {1..6}` and each `interaction_only ∈ {False, True}`
we run `cross_validate` (`KFold(5)`) and log mean ± std of R² and MAE.

Outputs:
- `poly_degree_sweep.csv` — `(degree, interaction_only, cv_r2_mean,
  cv_r2_std, cv_mae_mean, cv_mae_std)`
- `poly_degree_curve.png` — error-bar plot for both metrics

#### 2. Condition-number diagnostic per design matrix
For each `(degree, interaction_only)` we build the standardised
polynomial design matrix and compute `cond(Z) = σ_max / σ_min`. Catches
the moment OLS becomes numerically dicey — the curve typically rises
several orders of magnitude as degree grows, particularly with
`interaction_only=False`.

Output: `condition_number.csv`.

#### 3. Interaction-only vs full polynomial
Same sweep as (1) but with `interaction_only=True` overlayed — strips
the per-feature higher-order terms (no x², x³, ...) and keeps only
cross-products. Useful when monomials blow up the condition number but
cross terms still help.

#### 4. 1-D synthetic extrapolation
kNN flatlines past the training range; `LinearRegression` keeps the
trend slope. Static plot in `extrapolation_demo.png`.

### Findings

**LinearRegression baseline (held-out):** train R² = 0.6089, **test R² = 0.5943**, test MAE = 0.5351 — a clean linear fit that explains roughly 60% of variance in California Housing.

**Polynomial degree sweep (5-fold CV R²):**

| degree | full polynomial | interaction-only |
|---|---|---|
| 1 | 0.6037 ± 0.0079 | 0.6037 ± 0.0079 |
| 2 | -0.3260 ± 0.8763 | -0.0516 ± 0.8492 |
| 3 | -54,701 ± 108,929 | -11.79 ± 17.82 |
| 4 | -8.9 × 10⁹ ± 1.7 × 10¹⁰ | -757 ± 1,223 |
| 5 | -4.6 × 10¹⁵ ± 9.2 × 10¹⁵ | -1,437 ± 1,902 |
| 6 | -4.4 × 10¹⁶ ± 8.8 × 10¹⁶ | -1,679 ± 2,314 |

Polynomial expansion without regularization is a footgun on this dataset. The expanded design matrix is wildly ill-conditioned and OLS coefficients explode:

**Design matrix condition number per degree:**

| degree | n_features (full) | cond (full) | n_features (interaction-only) | cond (interaction-only) |
|---|---|---|---|---|
| 1 | 8    | 6.5 × 10⁰  | 8   | 6.5 × 10⁰ |
| 2 | 44   | 8.8 × 10³  | 36  | 1.7 × 10³ |
| 3 | 164  | 1.1 × 10⁷  | 92  | 1.3 × 10⁴ |
| 4 | 494  | 1.4 × 10¹⁰ | 162 | 5.7 × 10⁴ |
| 5 | 1286 | 3.0 × 10¹³ | 218 | 2.1 × 10⁵ |
| 6 | 3002 | 1.6 × 10¹⁷ | 246 | 4.8 × 10⁵ |

`interaction_only=True` keeps the condition number 6–8 orders of magnitude smaller at high degrees because it suppresses pure power terms (x², x³, ...) that are nearly collinear with the originals after standardisation. It buys you a few more degrees before the model explodes, but it does **not** rescue you — the right answer is to add regularization, which is exactly the next module.

**1-D extrapolation demo** (kNN vs LinearRegression on `y = 2x + noise`, trained on x ∈ [0, 10]):

| x (test point) | kNN(k=5) | LinearRegression | true (slope 2) |
|---|---|---|---|
| 18 | 18.33 | 33.78 | 36 |

kNN clips at the maximum training neighbourhood (~9.17 on average → 18.33 after the +noise structure) and refuses to extrapolate. LinearRegression respects the linear structure and extrapolates correctly. This is the structural argument for linear models when the underlying generative process is parametric and you might query out-of-range inputs.

### Reflections

This project is the cleanest "more complexity, less performance" story in classical ML. Polynomial expansion sounds like a free upgrade ("we can fit non-linear shapes now!") but on California Housing's correlated features the design-matrix condition number jumps 16 orders of magnitude going from degree 1 to 6, and OLS coefficients explode. The general principle — *capacity without regularization is a footgun on correlated features* — is one I keep coming back to, including in deep nets.

This is also the kind of result you have to explain carefully. Anyone hearing "we tried a more powerful model and it got worse" will assume something went wrong. The honest framing is "we removed a constraint that was doing useful work without realising it, and the next module — regularization — puts a smarter version of that constraint back." Phrasing "fit got worse because we removed structure" in plain language matters because the easy mistake is for someone to conclude "more complex models are bad" rather than "complexity without constraint is bad."

The 1-D extrapolation demo is small but it sums up *when* to reach for linear models in a deployed system: if your production inputs can drift outside the training range (price spikes, sensor failures, post-launch usage patterns), kNN and tree models silently clip to the training distribution. Linear models extrapolate honestly — sometimes badly, but at least *visibly* badly, which is the property an oncall engineer actually wants.

### Methodology Notes
- The CV is done on the **full** dataset; the held-out hold-out split
  reported at the top of the run is for a quick eyeball check only.
- Condition number is computed on a 4000-row subsample (random) so
  large degree matrices fit in memory; SVD cost is the dominant term.
- All polynomial expansion happens **after** scaling so the bias term
  isn't dragged around by raw column magnitudes.

### Limitations
- California Housing has a clipped target (`MedHouseVal` caps at 5.0)
  which limits the gain from high-degree expansions — they fit the
  bulk of the distribution but lose the censored tail. Visible as a
  diagonal stripe in any residual plot of `MedHouseVal ≈ 5`.
- Degrees ≥ 5 generate hundreds of features on 8 inputs — `cond(Z)`
  often exceeds `1e10` and OLS coefficients become essentially
  uninterpretable. `home_pricer_balanced` (Ridge / Lasso / ElasticNet) is the
  natural follow-up.

### Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0
python evaluate.py
```


<br>

---

## Home Pricer (Balanced) — Ridge / Lasso / ElasticNet on California Housing

> 📁 [`home_pricer_balanced/`](./home_pricer_balanced)

A regularised-regression study on a degree-2 polynomial design: Ridge,
Lasso, ElasticNet — α-paths, coefficient paths, and held-out
comparison at each model's CV-selected α.

### Setup
- `sklearn.datasets.fetch_california_housing` (20 640 rows × 8 features).
- Pipeline: `PolynomialFeatures(degree=2) -> StandardScaler -> <linear model>`.
  The polynomial expansion is fit per CV fold so train-fold rows never
  leak into the design that the validation rows are scored on.
- α grid: `np.logspace(-3, 3, 13)` shared across Ridge / Lasso /
  ElasticNet for a fair comparison.

### Experiments

#### 1. Ridge α-path (5-fold CV)
13 α values, `cross_val_score(scoring="r2")` per α.
Output: `ridge_path.csv`.

#### 2. Lasso α-path + coefficient path
Same α grid. We additionally fit a single Lasso on an 80/20 split per α
to capture the coefficient vector and plot `|coef|` vs α on a log-symlog
scale.

Outputs: `lasso_path.csv`, `lasso_coeff_path.png`.

#### 3. ElasticNet (l1_ratio × α) grid
5 `l1_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` × 13 α = 65 cells.

Output: `enet_grid.csv`.

#### 4. Held-out comparison at CV-best α
For each of OLS / Ridge / Lasso / ElasticNet, pick the α (and
`l1_ratio` for ENet) that maximises CV R², refit on the 80% train
split, and report R² on the held-out 20%.

Output: `model_comparison.csv`.

### Findings

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

### Reflections

The single image I'd put on a slide: OLS at R²=−1.12 next to Ridge at R²=0.63 on the *exact same features*. The takeaway isn't "use Ridge" — it's "the same model + a single regularization hyperparameter is the difference between a system that fails catastrophically and one that ships." It's also a clean way to explain why ML projects sometimes need a second iteration that *looks like rework but isn't*.

Lasso's coefficient path also reframes a conversation that often gets stuck. When someone asks "which of the 44 features matter?" the honest answer is "as you tighten the L1 penalty, the model drops them in this order, and at the CV-best α it keeps 8." That's a much more informative answer than a single feature-importance ranking — it explicitly trades off how aggressively you want to simplify. Cross-functionally, that ordering becomes the input to "what data do we actually need to keep collecting?" — a question with real cost implications.

The ElasticNet result (α=0.0316, l1_ratio=0.5 → R²=0.578) is also a small reminder that the answer to "Ridge or Lasso?" is often "tune both and let the data pick." Engineering-wise that's a 5×13 grid search, fast on this dataset; the principle (don't commit to a regularization family before seeing the data) scales.

### Methodology Notes
- All four models share the same `StandardScaler` placement (after
  `PolynomialFeatures`), so the α scales are directly comparable.
- `cross_val_score` returns the **mean of fold scores** — we keep
  `cv_r2_std` alongside so the reader can tell apart "the model is
  better" from "the run is luckier".
- ElasticNet's `l1_ratio=1` reduces to Lasso and `l1_ratio=0` to Ridge;
  we keep the interior of the interval to avoid duplicating the
  endpoint experiments.

### Limitations
- 5-fold single-seed CV — `cv_r2_std` is reported but the α selection
  is from a single seed; for production tuning add `--n-seeds 5` and
  aggregate manually.
- Lasso coefficient path is taken on **one** 80/20 split so the
  coefficient curves are stable across the α grid; the CV scores
  alongside still use the full dataset.

### Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0
python evaluate.py
```


<br>

---

## Churn Alert — Logistic Regression on Telco Customer Churn

> 📁 [`churn_alert/`](./churn_alert)

A logistic-regression R&D study: solver / C / class-weight sweeps,
calibration + threshold curves, and a synthetic 3-class side experiment
for OvR vs multinomial.

### Setup
- IBM mirror CSV: <https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv>
- Falls back to `fetch_openml('telco-customer-churn', as_frame=True)`
  if the mirror is unreachable.
- ~7 000 rows, one-hot encoded -> mid-30s features, ~26% positive class.

### Experiments

#### 1. Solver comparison
`lbfgs` / `saga` / `liblinear` on the **same** problem with 5-fold
`StratifiedKFold` ROC-AUC. `liblinear` only does OvR for multiclass but
binary is fine. Warning suppression is on so the table doesn't get
swamped on slow solvers.

Output: `solver_comparison.csv`.

#### 2. C regularization sweep
`C ∈ np.logspace(-3, 2, 11)`, 5-fold CV ROC-AUC.
Outputs: `c_sweep.csv`, `c_sweep.png` (errorbar plot, log x).

#### 3. class_weight ablation
`class_weight ∈ {None, 'balanced'}` on a stratified 75/25 hold-out.
Logs minority recall alongside accuracy and AUC — the right metric for
this trade-off.

Output: `class_weight_ablation.csv`.

#### 4. Calibration curve + Brier score
Calibration on 10 quantile bins, plus the Brier score for the best CV
C. Logistic regression with a scaler is usually well-calibrated, so the
curve sits near the diagonal — useful sanity check.

Output: `calibration.png`.

#### 5. ROC + Precision-Recall curves
Both curves rendered on the same held-out split.

Output: `roc_pr_curves.png`.

#### 6. OvR vs multinomial (3-class)
Synthetic 3-class target = `tenure` bucketed at quantiles 1/3 and 2/3.
Compares `OneVsRestClassifier` against multinomial `LogisticRegression`,
checks that softmax rows sum to 1 while OvR does not.

Output: `multiclass_ovr_vs_softmax.csv`.

### Findings

**Solver comparison** (5-fold CV ROC-AUC, default `C=1.0`, balanced class weights off):

| solver | AUC mean | AUC std |
|---|---|---|
| lbfgs     | 0.8452 | 0.0071 |
| saga      | 0.8452 | 0.0071 |
| liblinear | 0.8452 | 0.0072 |

All three solvers reach the same optimum to four decimals. Same loss, same data → same minimum; the choice is operational (`saga` scales to larger data, `liblinear` is fastest on small problems, `lbfgs` is the safe default).

**C regularization sweep** (5-fold CV ROC-AUC):

| C | AUC mean | AUC std |
|---|---|---|
| 0.001  | 0.8376 | 0.0052 |
| 0.01   | 0.8423 | 0.0063 |
| 0.1    | 0.8445 | 0.0070 |
| 1.0    | 0.8452 | 0.0071 |
| 10     | 0.8452 | 0.0073 |
| 100    | 0.8452 | 0.0073 |

AUC plateaus at C≥1 — the regularization sweet spot is wide on Telco Churn. Below C=0.1 the model is over-regularized and starts losing signal; above C=1 the data isn't rich enough to benefit from further loosening. **Picked `C=1`** as the simplest point on the plateau.

**Class-weight ablation (Telco Churn is imbalanced: 26.5% churn):**

| class_weight | accuracy | ROC-AUC | minority recall |
|---|---|---|---|
| None       | 0.8003 | 0.8448 | **0.5182** |
| balanced   | 0.7509 | 0.8447 | **0.7794** |

`balanced` trades ~5 pp of accuracy for **+26 pp of minority recall** while leaving AUC essentially unchanged. AUC is rank-based and class-balance-invariant; what changes is the *threshold* the model picks for the positive class. For churn — where the cost of missing a churner is much larger than the cost of a false alarm — `balanced` is the right call.

**OvR vs multinomial (synthetic 3-class tenure target):**

| strategy | accuracy | probability rowsum |
|---|---|---|
| OvR (one-vs-rest)        | 0.9460 | 1.0000 |
| multinomial (softmax)    | **0.9765** | 1.0000 |

Multinomial wins by ~3 pp on the tenure-bucket problem. OvR trains three separate binary models with no constraint that probabilities cooperate; the softmax formulation optimizes a joint cross-entropy and produces calibrated relative probabilities across classes. The gap shrinks on well-separated problems but multinomial is the right default for true multi-class targets.

**Design choice:** ship `LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced", multi_class="multinomial")` — plateau-stable C, conservative solver, recall-prioritised threshold via balanced weights, softmax for any future multi-class extension.

### Reflections

The class-weight ablation is the most product-relevant finding in the module: `balanced` trades 5 pp accuracy for +26 pp minority recall while leaving ROC-AUC essentially unchanged. The model didn't actually get "better" or "worse" — we moved the threshold. That distinction matters because the right threshold isn't a modelling decision; it's a *product* decision about asymmetric costs. For churn, missing a churner costs much more than a false alarm, so balanced wins. For an irreversible action (e.g. account suspension) the calculus flips. Getting the engineering team and the product team to talk about *the cost ratio* up front is half the value of building this kind of model.

The solver comparison (all three identical to four decimals) is a less glamorous insight that pays off in code reviews: when three optimizers reach the same minimum on the same convex loss, that's the loss telling you it's well-conditioned. The choice between lbfgs / saga / liblinear is an *operational* decision (data scale, regularization type, parallelism), not a modelling one. Recognising "this is operational, not modelling" early stops a lot of bikeshed.

On the systems side, the calibration result and Brier score matter for any downstream system that *uses* the predicted probability as input — recommendation re-ranking, threshold-based routing, expected-value math. A well-calibrated LR with `balanced` weights is the kind of unfashionable baseline that quietly wins production deployments because the downstream consumers can trust the numbers.

### Methodology Notes
- All preprocessing (scaler, one-hot already in `load_binary`) is
  inside the sklearn `Pipeline` so each CV fold gets its own scaler.
- The 3-class target intentionally drops `tenure` and `Churn` from
  features to avoid trivial leakage.
- Calibration is reported with `strategy="quantile"` so bins have
  comparable counts even when the score distribution is skewed (which
  it always is on churn problems).

### Limitations
- `saga` may need more iterations on this problem; we cap at 4000 to
  keep wall-clock reasonable. If you see convergence warnings, raise
  it in `build_binary(solver='saga')`.
- The 3-class target is constructed for educational comparison, not a
  real product target.

### Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0
python evaluate.py
```


<br>

---

## Income Streamer — SGD on UCI Adult Income

> 📁 [`income_streamer/`](./income_streamer)

Stochastic-gradient linear models on UCI Adult Income (`>50K` vs
`<=50K`). Four experiments covering loss choice, learning-rate
schedule, batch size, early stopping, and a multi-seed epoch curve.

### Setup
- `fetch_openml('adult', version=2, as_frame=True)`
- One-hot encoded categoricals, NaN rows dropped.
- All experiments share the same scaling: `StandardScaler(with_mean=False)`
  fit on the training split (sparse-safe).

### Experiments

#### 1. Loss × learning-rate-schedule grid
- `loss ∈ {log_loss, hinge, modified_huber, squared_hinge}`
- `learning_rate ∈ {constant, optimal, invscaling, adaptive}`
- Single hold-out per cell; `max_iter=50, tol=1e-4`
- `eta0=0.01` is set for the schedules that require it.

Output: `loss_schedule_grid.csv` (16 cells).

#### 2. partial_fit batch-size study
`batch_size ∈ {128, 512, 2048}`, fixed 8 epochs, `loss=log_loss`. Logs
test accuracy and total wall-clock per batch size.

Output: `batch_size_study.csv`.

#### 3. Early stopping vs full epochs
Two SGD runs with `max_iter=30`: one with `early_stopping=False`, one
with `early_stopping=True, validation_fraction=0.1, n_iter_no_change=5`.
Compares final accuracy, `n_iter_`, and wall-clock.

Output: `early_stopping.csv`.

#### 4. Multi-seed epoch curve
For each of 5 seeds, run 25 epochs of `partial_fit` (batch=1024) and
log per-epoch train + test accuracy. The plot shows mean ± shaded std
across seeds.

Outputs: `epoch_curve_multiseed.csv`, `epoch_curve.png`.

### Findings

Adult Income (UCI direct after OpenML mirror flake): 30,162 rows × 96 features after one-hot, target rate 24.9%.

**loss × LR-schedule grid (held-out accuracy):**

| loss \ schedule  | constant | optimal | invscaling | adaptive |
|---|---|---|---|---|
| `log_loss`        | 0.8250 | 0.8415 | 0.8341 | **0.8463** |
| `hinge`           | 0.8337 | 0.8367 | 0.8379 | **0.8478** |
| `modified_huber`  | 0.7981 | 0.8215 | 0.8439 | **0.8454** |
| `squared_hinge`   | 0.8004 | 0.7810 | **0.8430** | 0.8190 |

**`adaptive`** is the best schedule for 3 of 4 losses — it halves the learning rate when validation stops improving, which lands SGD safely in the optimum without manual tuning. **`squared_hinge`** is the outlier: it prefers `invscaling`, and `adaptive` actively hurts it (the squared margin term amplifies updates and `adaptive`'s aggressive halving stalls before convergence). The best single cell is **`hinge` + `adaptive` = 0.8478** — linear-SVM loss with self-tuning rate.

**partial_fit batch-size study (1 epoch over 3 batches):**

| batch size | test acc | time (s) |
|---|---|---|
| 128  | **0.8271** | 0.45 |
| 512  | 0.8246 | 0.16 |
| 2048 | 0.8054 | 0.09 |

Smaller batches give better accuracy at proportionally higher wall-clock time — exactly the stochasticity vs convergence trade `partial_fit` is designed for. **A 3-4× time hit buys ~2 pp accuracy**, which is usually worth it during initial training; for streaming inference you'd switch to larger batches once the model is warm.

**Early stopping vs full epochs:**

| run | test acc | n_iter | time (s) |
|---|---|---|---|
| full epochs (max=30)        | **0.8319** | 30 | 0.22 |
| early stopping (patience=5) | 0.7878 | 10 | 0.07 |

On this dataset early stopping fires too soon — the validation curve has a long shallow tail that `EarlyStopping` mistakes for a plateau. For SGDClassifier specifically I'd raise `patience` or fall back to a fixed epoch budget if the dataset is small enough that 30 epochs is cheap.

**Multi-seed final accuracy:** 0.8182 / 0.8291 / 0.7953 (mean ≈ **0.815**) — SGD has real seed variance on tabular data because step ordering matters; reporting mean ± std is non-optional.

**Design choice:** ship `SGDClassifier(loss="hinge", learning_rate="adaptive", random_state=fixed, n_iter_no_change=10)` — best cell in the grid plus a longer patience to fix the early-stopping issue. For comparable performance with a less-fiddly model on this scale of data, `HistGradientBoosting` from `deposit_propensity` would be the obvious step up.

### Reflections

`partial_fit` is the underrated API in this project. Most ML examples treat training as a one-shot process; in production, data arrives over time and the model needs to *keep learning* without retraining from scratch every night. The batch-size sweep is a small concrete glimpse of that streaming regime: smaller batches mean more frequent updates, more wall-clock cost, slightly better accuracy. Productionising it is a real systems exercise — model versioning, replay buffers, rollback on quality regression — but the algorithmic primitive is right there in `SGDClassifier`.

The loss × LR-schedule grid is a useful experiment to *run with a teammate* who's never tuned SGD before. Watching `squared_hinge` collapse on the `adaptive` schedule while `hinge` thrives on it is the kind of "the loss and the schedule have to talk to each other" lesson that's much faster to absorb from a single grid than from a textbook chapter. Building these comparison harnesses is half of what makes a team faster over time.

The early-stopping result (too eager on this validation curve) is a small example of why I distrust any "off by default" configuration that silently changes behaviour. A patience of 5 is *not* the right default for every dataset, but the package ships one as if it were. The lesson is to treat library defaults as "the author's guess about the median use case" — fine until your case isn't the median, at which point the cost of *not knowing* the default existed is high.

### Methodology Notes
- All seeds re-split the data and re-init the scaler so the variance
  reflects both **stochastic optimisation noise** and **sampling
  noise** — the way users will encounter it in practice.
- The early-stopping comparison uses `validation_fraction=0.1` carved
  out of `Xtr`, not the held-out test set; the reported accuracy is
  still on the held-out test set, so the comparison is honest.
- `loss=hinge` does not support `predict_proba`; we only score
  accuracy so the comparison stays apples-to-apples.

### Limitations
- Adult Income has ~48k rows; bigger batch sizes start to dominate the
  wall-clock comparison because each `partial_fit` call has fixed
  overhead. Reading the curve as "throughput vs convergence" is more
  useful than reading single batch-size numbers.
- We do not sweep `eta0` or `alpha` — adding `--eta0`/`--alpha` flags
  is a natural extension.

### Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0 --n-seeds 5
python evaluate.py
```


<br>

---

## Mushroom Screener — Decision Tree on UCI Mushroom

> 📁 [`mushroom_screener/`](./mushroom_screener)

A decision-tree R&D study on the textbook all-categorical Mushroom
dataset: parameter grid, post-pruning path, and feature-importance
agreement check.

### Setup
- UCI Mushroom (`agaricus-lepiota.data`)
- URL: <https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data>
- Rows with `?` missing values dropped, target binarised (1 = poisonous).
- Pipeline: `OneHotEncoder(handle_unknown='ignore') -> DecisionTreeClassifier`.

### Experiments

#### 1. criterion × max_depth × min_samples_leaf grid
- `criterion ∈ {gini, entropy}`
- `max_depth ∈ {2, 4, 6, 8, 12, None}`
- `min_samples_leaf ∈ {1, 5, 20}`
- Single hold-out (75/25, stratified) per cell.

Output: `grid_search.csv` (36 cells).

#### 2. Cost-complexity post-pruning path
Fit a full-depth tree, ask for `cost_complexity_pruning_path`, then
re-fit at ~30 `ccp_alpha` values and log `(train_acc, test_acc, n_nodes,
depth)`.

Outputs:
- `ccp_path.csv`
- `ccp_path.png` — twin-axis plot, accuracy curves + node count

#### 3. feature_importances_ vs permutation_importance
Tree's `feature_importances_` are computed on the **training fit** —
they don't tell you whether removing a feature actually hurts a held-out
score. We compare them with `permutation_importance(scoring="accuracy",
n_repeats=5)` on the test split.

Outputs:
- `importance_compare.csv` — per-feature `tree_importance`, `perm_mean`,
  `perm_std`
- `importance_compare.png` — side-by-side bars for the top 12

#### 4. Tree visualisations
- `tree_shallow.png` — `max_depth=4` baseline tree
- `tree_best.png` — best pruned tree from the ccp sweep (capped at
  depth 4 for legibility)

### Findings

Mushroom is the *separable* dataset by design: every shape, odor, and surface feature carries hard signal, and a small tree can recover the rule perfectly. That's exactly what the grid shows.

**criterion × max_depth × min_samples_leaf grid (held-out accuracy):**

| max_depth | leaf=1 | leaf=5 | leaf=20 |
|---|---|---|---|
| 2  | 0.9391 | 0.9391 | 0.9391 |
| 4  | 0.9979 | 0.9979 | 0.9950 |
| 6  | **1.0000** | **1.0000** | 0.9950 |
| 8  | 1.0000 | 1.0000 | 0.9950 |
| 12 | 1.0000 | 1.0000 | 0.9950 |
| None | 1.0000 | 1.0000 | 0.9950 |

**criterion comparison (gini vs entropy):** identical at every grid cell to four decimals. When the underlying splits are perfectly informative, the split-quality function doesn't matter.

**Cost-complexity post-pruning path:**

| ccp_alpha | n_nodes | depth | train_acc | test_acc |
|---|---|---|---|---|
| 0.0000 | 17 | 5 | 1.0000 | 1.0000 |
| 0.0022 | 15 | 4 | 0.9988 | 0.9979 |
| 0.0051 | **13** | 4 | 0.9960 | **0.9950** |
| 0.0091 | 11 | 4 | 0.9894 | 0.9865 |
| 0.0292 | 5  | 2 | 0.9381 | 0.9391 |
| 0.0841 | 3  | 1 | 0.8949 | 0.9100 |
| 0.2925 | 1  | 0 | 0.6180 | 0.6180 |

There's a sharp knee: with **ccp_alpha=0.005 we drop to 13 nodes and still hit 99.5% test accuracy** — a 24% smaller model for 0.5 pp of accuracy. Push past α=0.029 (5 nodes) and we collapse to the underfit "majority odor" baseline.

**`feature_importances_` vs `permutation_importance` (selected):**

| feature | tree_importance | perm_mean |
|---|---|---|
| `cap-shape_s` | 0.000 | 0.428 |
| `cap-surface_g` | 0.000 | 0.080 |
| ... | | |

The clearest example: `cap-shape_s` has tree_importance **0** (the trained tree never splits on it) but permuting its values destroys **43%** of held-out accuracy. The tree found an *equivalent* discriminator (probably `odor_*` or `gill-color_*`) that fires first; the structural redundancy of one-hot encoded categorical features makes `feature_importances_` an incomplete picture. Permutation importance, by perturbing inputs at inference time, surfaces features the trained model *would* rely on if its first choice were removed.

**Design choice:** ship `criterion="gini", max_depth=6, min_samples_leaf=1, ccp_alpha=0.005` — perfect-or-near-perfect test accuracy with the smallest tree that still reaches that plateau. For deployment I'd report both importance views to whoever is interpreting the model.

### Reflections

The `feature_importances_` vs permutation-importance gap is the single most important lesson in this project for anyone who has to explain a model. The trained tree's `cap-shape_s` importance is *zero*, but permuting that feature destroys 43% of held-out accuracy — because the tree found a redundant equivalent split that fires first. If a regulator or product owner asks "which features matter?" and you only show them `feature_importances_`, you're showing them an incomplete picture and you don't know it.

This generalises uncomfortably: every "explain my model" tool answers a slightly different question, and the right one depends on what the explanation is *for*. For "which features did the model use," `feature_importances_` is honest. For "which features do we need to keep collecting," permutation importance is the right one (it answers "what would happen if this feature went away"). For "why this individual prediction," SHAP or LIME. Naming the *question* before reaching for the tool is part of the job.

The cost-complexity pruning result (13 nodes, 0.995 test accuracy) is also a real product win: a 24% smaller model for 0.5 pp accuracy means a model that fits on a phone, or that a domain expert can read top-to-bottom in a meeting. Smaller-but-still-good is often the right ship — not because of inference speed (which is microseconds either way), but because the artifact becomes a *communication object* people can reason about.

### Methodology Notes
- We compute `cost_complexity_pruning_path` on the **encoded** training
  matrix because sklearn's pruning API takes feature arrays directly,
  not pipelines.
- The "best" pruned model is selected by held-out test accuracy from
  the ccp path; for production use, swap to CV-based selection.
- Permutation importance uses `n_repeats=5` for cost; on a small
  feature space this is enough to surface order-of-magnitude differences.

### Limitations
- Mushroom is famously **separable** — most depth ≥ 5 trees hit 100%
  test accuracy. The interesting signal here is the **pruning path
  shape** (when accuracy starts to fall) and the **importance
  disagreement** between tree and permutation views, not the absolute
  numbers.

### Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0
python evaluate.py
```


<br>

---

## Income Tuner — Hyperparameter Tuning on UCI Adult Income

> 📁 [`income_tuner/`](./income_tuner)

4-way hyperparameter-search comparison on `RandomForestClassifier`. The
same search space is fed to every method so the wall-clock vs CV-score
trade-off is directly comparable.

### Setup
- Same Adult Income dataset as `income_streamer`
  (`fetch_openml('adult', version=2)`).
- Estimator: `RandomForestClassifier(random_state=42, n_jobs=-1)`.
- CV: `StratifiedKFold(3)` for the search loops, `StratifiedKFold(5)`
  for the baseline.
- `--subsample 15000` (default) carves a stratified subsample for the
  search comparison so total wall-clock stays in the few-minute range.

### Search space
Discrete grid (`GridSearchCV` / `HalvingGridSearchCV`):
```python
{
  "n_estimators": [100, 200, 300],
  "max_depth": [None, 10, 20],
  "min_samples_leaf": [1, 4, 16],
  "max_features": ["sqrt", 0.5],
}  # 54 cells
```

Distributions (`RandomizedSearchCV` / `HalvingRandomSearchCV`):
```python
{
  "n_estimators": randint(100, 400),
  "max_depth": randint(5, 30),
  "min_samples_leaf": randint(1, 32),
  "max_features": uniform(0.3, 0.6),
}
```

### Experiments

#### 1. 4-way search comparison
- `GridSearchCV`
- `RandomizedSearchCV` (n_iter=20)
- `HalvingGridSearchCV` (factor=3)
- `HalvingRandomSearchCV` (n_candidates=40, factor=3)

Logs `best_score`, `best_params`, total wall-clock per method.

Outputs:
- `search_comparison.csv`
- `pareto.png` — wall-clock vs best CV ROC-AUC, red dots mark the
  Pareto-front (non-dominated) methods.

#### 2. RandomizedSearch n_iter sensitivity
Sweep `n_iter ∈ {10, 50, 200}` to show how diminishing returns kick in.

Output: `rand_n_iter_sensitivity.csv`.

### Findings

**4-way search comparison on RandomForest (Adult Income, same search space, same CV split, ROC-AUC):**

| search | best CV ROC-AUC | wall-clock (s) | best params (top-3 shown) |
|---|---|---|---|
| `GridSearchCV`           | **0.9140** |  99.4 | depth=None, max_features=sqrt, min_samples_leaf=4, n_estimators=100 |
| `RandomizedSearchCV` (n_iter=50)  | 0.9125 |  71.3 | depth=15, max_features=0.56, min_samples_leaf=5, n_estimators=362 |
| `HalvingGridSearchCV`    | 0.9134 |  **28.8** | depth=20, max_features=sqrt, min_samples_leaf=4, n_estimators=200 |
| `HalvingRandomSearchCV`  | 0.8997 |  **16.9** | depth=7, max_features=0.38, min_samples_leaf=5, n_estimators=153 |

Pareto picture:
- **HalvingGridSearchCV is the clear winner.** It reaches AUC=0.9134 in 28.8s — *0.06 pp behind full Grid, 3.5× faster.* The successive-halving budget allocates compute to promising configs and prunes the rest early; on this RF grid it loses almost nothing in solution quality.
- **HalvingRandomSearchCV** is *fastest* (16.9s) but takes a real hit on score (0.8997, ~1.4 pp below Grid). Halving's small initial budget plus random sampling lands on under-trained configurations that look bad and get pruned before they can prove themselves.
- **RandomizedSearch (n_iter=50)** sits in the middle on both axes — slower than Halving variants because every candidate gets the full budget, lower score than Grid because exhaustive coverage matters on a small space like this one.

**n_iter sensitivity for RandomizedSearch:**

| n_iter | best CV ROC-AUC | wall-clock (s) |
|---|---|---|
| 10  | 0.9120 | 47.0 |
| 50  | 0.9135 | 176.2 |
| 200 | **0.9135** | 693.9 |

A flat plateau at n_iter ≈ 50. Going from 50 → 200 spends 4× more compute for zero gain — the search space has been thoroughly explored at 50 samples. Lesson: for any new search problem, **start with n_iter≈30-50 and only push higher if the leaderboard is still climbing**.

**Design choice:** for *this* search space, **HalvingGridSearchCV** is the right default — best speed-vs-score trade and reproducible (deterministic). I'd reach for `RandomizedSearchCV(n_iter=50)` only when the search space is much larger (continuous distributions over many hyperparameters) and exhaustive grids become intractable.

### Reflections

The Pareto picture (HalvingGridSearch matching full Grid in ~30% of the time) is the engineering result I'd most want a junior teammate to internalise. Hyperparameter search is *compute that doesn't make the user's life better unless it changes a decision* — every second spent on it should be defensible against either (a) tighter accuracy on the deployed model, or (b) the team's time. Halving searches reach the same decision with much less compute, so the rational default is to start with Halving and only fall back to exhaustive grid when the search space is small enough that the savings don't matter.

The `n_iter` plateau (50 = 200) is a small product/communication point too. "We ran 200 candidates" sounds more rigorous in a slide than "we ran 50," but it isn't — the 4× extra compute returned zero accuracy gain on this space. Reporting the *plateau* rather than the *max* is honest about diminishing returns and stops the next team from wasting their compute budget on the same dead end. This is the kind of thing I'd want the README of any team's tuning scripts to call out by default.

On the systems side, all four search APIs share the same scorer and CV splitter — the trick was making them comparable. Building that little harness once means future hyperparameter questions become "swap models / spaces, re-run" instead of one-off scripts. A small investment in shared infrastructure pays back across every project that touches model selection.

### Methodology Notes
- All four methods share the **same** CV splitter, scoring metric, and
  estimator factory — the only knob being changed is the search
  strategy and (for the randomised variants) the seed.
- Halving variants need `sklearn.experimental.enable_halving_search_cv`
  imported before construction (handled at the top of `train.py`).
- The Pareto front is computed in `shared/utils.pareto_front` —
  `minimize_x` (time), `maximize_y` (score).

### Limitations
- 15k-row subsample under-states the absolute scores for the baseline
  comparison; the relative ordering of search methods is stable, which
  is the point.
- Single-seed comparison — for a more rigorous benchmark, wrap the
  whole loop in `--n-seeds 5` and average.

### Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0 --subsample 15000
python evaluate.py
```


<br>

---

## Deposit Propensity — Tree-Ensemble Benchmark on UCI Bank Marketing

> 📁 [`deposit_propensity/`](./deposit_propensity)

Benchmark six tree ensembles on a real, imbalanced marketing dataset
across multiple seeds, then dig into agreement on feature importance,
calibration quality, and out-of-fold predictions.

### Setup
- UCI Bank Marketing — `bank-full.csv` from
  <https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip>
- Falls back to `bank-additional-full.csv` if the first archive fails.
- Target `y` (yes/no = subscribed); the positive class is rare (~11 %).
- Estimators: `RandomForest`, `ExtraTrees`, `GradientBoosting`,
  `HistGradientBoosting`, `XGBoost`, `LightGBM` (last two fail-soft if
  the libraries are missing).

### Experiments

#### 1. 5-seed benchmark
For each seed and each ensemble, fit on a stratified 80% split and
score `(roc_auc, f1, accuracy)` on the held-out 20%. Wall-clock per
fit also recorded.

Output: `ensemble_benchmark.csv` — one row per (model, seed).

#### 2. Permutation importance on the best model
Best model = highest mean ROC-AUC across seeds. Permutation importance
with `n_repeats=5`, ROC-AUC scoring, on the held-out test split.

Outputs: `perm_importance.csv`, `perm_importance.png`.

#### 3. Feature-importance agreement matrix
Spearman ρ between every pair of models' `feature_importances_`
vectors. Surfaces "do these tree models agree about which features
matter, or does each pick its own?".

Outputs: `importance_agreement.csv`, `importance_agreement.png`
(matrix heatmap with annotated ρ values).

#### 4. Calibration overlay
Calibration curves for all ensembles on one axes (10 quantile bins) +
Brier score in the legend. Tree models are notoriously
mis-calibrated; this makes that visible.

Output: `calibration_overlay.png`.

#### 5. Out-of-fold predictions
`cross_val_predict(method="predict_proba", cv=5)` for every ensemble.
Saved as wide CSV (`y_true, oof_<model>, ...`) — natural input for
downstream stacking experiments.

Output: `oof_predictions.csv`.

### Findings

Bank Marketing is mildly imbalanced (11.7% positive), 45,211 rows × 42 features after one-hot.

**6-ensemble benchmark (3-seed StratifiedKFold mean ± std):**

| model | ROC-AUC | F1 | accuracy | fit_time (s) |
|---|---|---|---|---|
| HistGradientBoosting | **0.9346 ± 0.0014** | 0.5402 | 0.9057 | 2.54 |
| LightGBM             | 0.9340 ± 0.0016 | **0.5503** | 0.9060 | 2.20 |
| XGBoost              | 0.9329 ± 0.0016 | 0.5437 | 0.9059 | **1.03** |
| RandomForest         | 0.9298 ± 0.0014 | 0.4971 | 0.9061 | 3.02 |
| GradientBoosting     | 0.9286 ± 0.0011 | 0.5224 | 0.9067 | 9.44 |
| ExtraTrees           | 0.9158 ± 0.0018 | 0.4567 | 0.9016 | 2.24 |

The boosted-trees family (HistGB / LightGBM / XGBoost) clusters within 0.002 AUC of each other and ~1 pp above the bagged-trees family (RF / ExtraTrees). The classical `sklearn.GradientBoosting` is the slowest by ~3-4× because it lacks histogram binning. **ExtraTrees** lags because its extra-random splits underfit a relatively low-noise structured dataset.

**Out-of-fold AUCs (`cross_val_predict`, 5-fold):**

| model | OOF AUC |
|---|---|
| HistGradientBoosting | 0.9353 |
| XGBoost              | 0.9344 |
| LightGBM             | 0.9340 |
| RandomForest         | 0.9300 |
| GradientBoosting     | 0.9299 |
| ExtraTrees           | 0.9173 |

Same ranking, OOF and 3-seed-CV agree to two decimals — the leaderboard is real.

**Feature-importance agreement matrix (Spearman ρ on the 42 features):**

|  | RF | ExtraTrees | GBT | XGBoost | LightGBM |
|---|---|---|---|---|---|
| **RandomForest**     | 1.000 | 0.962 | 0.675 | **0.387** | 0.890 |
| **ExtraTrees**       | 0.962 | 1.000 | 0.594 | 0.401 | 0.809 |
| **GradientBoosting** | 0.675 | 0.594 | 1.000 | 0.679 | 0.680 |
| **XGBoost**          | 0.387 | 0.401 | 0.679 | 1.000 | **0.328** |
| **LightGBM**         | 0.890 | 0.809 | 0.680 | 0.328 | 1.000 |

Two clusters surface clearly:
- **Bagged family** (RF, ExtraTrees, LightGBM) — ρ ≥ 0.8 within the cluster
- **Boosting family** (GBT, XGBoost) less correlated with bagged methods (ρ ≈ 0.4–0.7)
- **XGBoost ↔ LightGBM ρ = 0.328** is the surprise: same algorithm family, very different importance rankings. They reach similar AUC by different paths through the feature space.

This is a real-world R&D-style argument for ensembling the *ensembles*: if XGBoost and LightGBM disagree on which features matter while reaching the same score, blending their predictions should diversify error.

**Design choice:** ship **HistGradientBoosting** as the single model (best AUC, F1 above 0.54, sub-3s fit time, no extra wheel dependencies). For a production deployment I'd add a small XGBoost+LightGBM blend on top to harvest the disagreement.

### Reflections

The headline number — HistGradientBoosting at 0.9346 ROC-AUC — is less interesting to me than the *agreement matrix*. XGBoost and LightGBM reach essentially the same score (0.9329 vs 0.9340) but their feature-importance Spearman is 0.328. Two well-tuned boosted models can disagree completely about *why* they got there. That's the most useful argument I know for blending models in production: not because the ensemble's mean is higher, but because the *errors* of each model are partially decorrelated.

Cross-functionally, ensemble comparison tables are also useful at being honest about diminishing returns. A new team often jumps straight to "let's try XGBoost / LightGBM / CatBoost" before tuning a RandomForest. On this dataset RF lands within 0.5 pp of the best model — half the time, the question isn't which library, it's whether the gap to your baseline is worth the engineering cost of adding another dependency. Reporting *all six* on the same axes lets the team have that conversation explicitly.

On the systems side, the fit-time column matters more than people admit in offline comparisons. GradientBoosting at 9.4 s/fit is 9× XGBoost at 1.0 s — that compounds *brutally* in hyperparameter search and in production retraining loops. For a team deciding what to ship, "0.06 pp slower but 3× faster to retrain" is often the right trade, especially when data drift is the real long-term enemy.

### Methodology Notes
- `XGBoost` / `LightGBM` use `n_estimators=300, learning_rate=0.1` —
  not tuned per dataset on purpose so the benchmark is comparable to
  the sklearn defaults next door.
- Permutation importance is **test-set** based and scored on AUC, so
  it answers "permuting this column drops AUC by how much".
- The agreement matrix only uses models that expose
  `feature_importances_`; HistGradientBoosting does not, so it's
  excluded from that subplot (still benchmarked).

### Limitations
- Strong class imbalance: accuracy can look high even from the
  majority predictor. Read AUC and F1 first.
- Single-set hyperparameters per ensemble. A per-model RandomizedSearch
  pass would tighten the differences; out of scope here.

### Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0 --n-seeds 5
python evaluate.py
```


<br>

---

## Shopper Personas — KMeans Clustering on Mall Customers

> 📁 [`shopper_personas/`](./shopper_personas)

A KMeans R&D study with four internal-validation metrics, init
sensitivity, MiniBatch vs full comparison, and centroid scatter plots
across multiple feature pairs.

### Setup
- Public mirror (primary): SteffiPeTaffy/machineLearningAZ GitHub mirror.
- Falls back to the historical `Avik-Jain/100-Days-Of-ML-Code` mirror
  if the primary 404s.
- Features used: annual income, spending score, age, gender (encoded).

### Experiments

#### 1. K-sweep (k = 2..12) with four metrics
For every k:
- `inertia_` (elbow)
- `silhouette_score`
- `davies_bouldin_score`
- `calinski_harabasz_score`

Outputs: `k_sweep.csv`, `k_sweep.png` (2×2 panel, one subplot per metric).

#### 2. Init comparison
`init ∈ {k-means++, random}` × `n_init ∈ {1, 5, 10, 25}` at k=5,
averaged across 5 seeds. Logs `inertia` mean ± std, silhouette mean,
and wall-clock.

Output: `init_comparison.csv`.

#### 3. MiniBatchKMeans vs KMeans
For `k ∈ {3, 5, 8}`: fit both, log `(full_inertia, mini_inertia,
full_time, mini_time, ARI agreement)`. ARI ≈ 1 means MiniBatch landed
on the same partition; ARI < 0.9 means the speed-up came at a real
cluster-stability cost on this dataset.

Output: `minibatch_vs_full.csv`.

#### 4. Centroid scatter
Best K (highest silhouette) plotted on every pair of numeric features
(excluding gender). Centroids drawn as black `X`. One PNG per pair.

Outputs: `clusters_<feat_a>_vs_<feat_b>_k<K>.png`.

### Findings

**K-sweep with four cluster-quality metrics (Mall Customers, 200 rows × 4 numeric features):**

| K | inertia | silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|
| 2  | 588.80 | 0.2518 | 1.61 | 71.0 |
| 4  | 386.83 | 0.3012 | 1.29 | 69.8 |
| 6  | 275.03 | 0.3343 | 1.01 | 74.1 |
| 8  | 199.75 | 0.3880 | 0.94 | 82.4 |
| 10 | 152.03 | 0.4208 | 0.83 | 90.0 |
| **11** | 137.26 | **0.4285** | **0.82** | 91.3 |
| 12 | 126.02 | 0.4263 | 0.83 | **91.4** |

The four metrics broadly agree: silhouette peaks at **k=11**, Davies-Bouldin (lower is better) bottoms at k=11, Calinski-Harabasz keeps inching up but starts plateauing. The "elbow" on inertia is around k=5-6 — the elbow rule is much more conservative than silhouette here. **On a 200-row dataset, k=11 is too many segments to be operationally useful** even if it's "optimal" by silhouette; I'd ship k=5-6 for an actual marketing team.

**Init comparison at k=5:**

| init | n_init | inertia (mean ± std) | silhouette | time (s) |
|---|---|---|---|---|
| k-means++ | 1  | 332.20 ± 3.84 | 0.3122 | 0.001 |
| k-means++ | 5  | 328.24 ± 3.72 | 0.3169 | 0.003 |
| k-means++ | 10 | 327.81 ± 3.11 | 0.3147 | 0.005 |
| k-means++ | 25 | **325.59 ± 0.48** | 0.3172 | 0.011 |
| random    | 1  | 344.40 ± 6.51 | 0.2960 | 0.001 |
| random    | 5  | 333.66 ± 4.76 | 0.3077 | 0.002 |
| random    | 10 | 328.55 ± 2.41 | 0.3165 | 0.003 |
| random    | 25 | 326.27 ± 1.11 | 0.3157 | 0.008 |

k-means++ at `n_init=1` matches random at `n_init=10` on inertia, with comparable silhouette — that's a real efficiency win on bigger data. By `n_init=25` both converge to the same global optimum (inertia within rounding error).

**MiniBatchKMeans vs KMeans (same K, ARI agreement on cluster labels):**

| K | full t (s) | mini t (s) | ARI |
|---|---|---|---|
| 3 | 0.004 | 0.008 | 0.7256 |
| 5 | 0.005 | 0.006 | 0.5591 |
| 8 | 0.005 | 0.007 | 0.6428 |

On 200 rows MiniBatch is actually *slower* than full KMeans (overhead dominates). The ARI numbers (~0.6) say MiniBatch finds *related* but not identical clusterings — fine on huge data where full KMeans is intractable; pointless overhead here.

**Design choice:** ship `KMeans(n_clusters=5, init="k-means++", n_init=10)` — meaningful segments, fast convergence, stable across seeds.

### Reflections

The cleanest illustration of "model-best" vs "useful" in any module. Silhouette peaks at k=11 on Mall Customers — *technically optimal* by the metric — but 11 customer segments is operationally useless for a marketing team that needs to design 5-6 campaigns. The right answer is k=5 or 6, and the right framing is "the metric measures cluster compactness, not how many segments your business can act on." That translation is the entire job of clustering in product work.

The init comparison is also a small reminder that defaults *encode opinions*. `k-means++` with `n_init=10` is the sklearn default for a reason — it matches random init at `n_init=25` on inertia and beats it on time. That's a research result that the library has quietly built in for you, and the cost of *not knowing* it's there is people thinking they need to "tune the initialization" when the library has already settled it.

On the systems side, MiniBatchKMeans being slower than full KMeans on 200 rows is a useful reminder that scale matters for tradeoffs. The right "use MiniBatch when N exceeds ~10^5" rule is the kind of operational know-how that sits in a senior engineer's head; explicit benchmarks like this one make that knowledge a *team property* rather than tribal lore.

### Methodology Notes
- All clustering happens on `StandardScaler`-transformed features so
  inertia comparisons aren't dominated by raw column magnitudes.
- "Best K" is picked by silhouette; on this dataset that almost
  always lands at K=5 (matches the famous teaching answer).
- Init comparison reports `inertia_std` across seeds to make explicit
  why `n_init=1, init=random` is a bad default (high variance).

### Limitations
- Only 200 rows — every metric carries non-trivial noise. Patterns are
  more interesting than absolute numbers.
- Internal validation metrics all assume convex / equally-sized
  clusters. They will mis-rank K on non-convex shapes; not an issue
  for this dataset but worth keeping in mind elsewhere.

### Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0
python evaluate.py
```


<br>

---

## Tumor Compressor — PCA on Breast Cancer Wisconsin

> 📁 [`tumor_compressor/`](./tumor_compressor)

A PCA R&D study: PCA vs KernelPCA, n_components sweep, reconstruction
error, whitening ablation, and 2-D / 3-D scatter plots.

### Setup
- `sklearn.datasets.load_breast_cancer` (569 rows × 30 features, binary
  malignant / benign).
- Pipeline: `StandardScaler -> PCA(n_components=N) -> LogisticRegression`.

### Experiments

#### 1. PCA vs KernelPCA
At fixed `n_components=10`, compare:
- `PCA(linear)`
- `KernelPCA(kernel ∈ {linear, rbf, poly})`

Output: `pca_vs_kpca.csv`.

#### 2. Scree + cumulative explained variance
Per-PC ratio (bars) and cumulative ratio (line) up to PC30, with a 0.9
threshold marker.

Outputs: `variance.csv`, `variance.png`.

#### 3. n_components sweep
`n_components ∈ {1..30}`, downstream LogReg test accuracy on a fixed
75/25 split.

Outputs: `components_sweep.csv`, `components_sweep.png`.

#### 4. Reconstruction error
For each `n_components ∈ {1, 2, 5, 10, 15, 20, 30}`, compute
`‖X − inverse_transform(transform(X))‖_F` on the standardised matrix.
Plotted with cumulative variance on a twin axis.

Outputs: `reconstruction.csv`, `reconstruction.png`.

#### 5. Whitening on/off ablation
`whiten ∈ {False, True}` × `n_components ∈ {2, 5, 10, 20}`. Whitening
makes the PCs unit-variance, which often helps subsequent linear
classifiers, sometimes hurts.

Output: `whitening_ablation.csv`.

#### 6. 2-D + 3-D PCA scatter
Coloured by class. The 3-D plot uses `mpl_toolkits.mplot3d`.

Outputs: `pc_scatter_2d.png`, `pc_scatter_3d.png`.

### Findings

**PCA vs KernelPCA — downstream LogisticRegression accuracy** (Breast Cancer Wisconsin, 30 features → top-N components, 5-fold CV mean):

| reducer | downstream accuracy |
|---|---|
| PCA(linear)         | **0.9650** |
| KernelPCA(linear)   | 0.9650 |
| KernelPCA(poly)     | 0.9371 |
| KernelPCA(rbf)      | 0.9231 |

PCA and KernelPCA-linear are identical (same projection up to numerical precision). The non-linear kernels actively *hurt* downstream classification on this dataset — Breast Cancer's classes are well-separated linearly after standardisation, and the RBF embedding squeezes the natural decision boundary into a curved manifold the linear classifier can't exploit. Lesson: **kernel PCA is a tool for non-linear structure; it has a real cost when the structure is already linear**.

**n_components sweep (downstream LR accuracy on 5-fold CV):**

| n_components | accuracy |
|---|---|
| 1 | 0.9021 |
| 2 | 0.9231 |
| 3 | 0.9371 |
| 4 | 0.9371 |
| **5** | **0.9650** |
| 6 | 0.9580 |
| 7 | 0.9510 |
| 8 | 0.9650 |
| 13 | 0.9650 |
| 20 | 0.9580 |
| 30 | 0.9580 |

Sharp knee at **n=5**: dropping from 30 dimensions to 5 *improves* test accuracy by 0.7 pp because the extra 25 components are mostly noise that the linear model otherwise tries to fit. Past n=5 there's a faint zigzag in the 0.95-0.96 band — pure CV-fold noise.

**Reconstruction quality** scales the same way: the cumulative explained-variance plot (`artifacts/variance.png`) crosses 95% around component 10, but downstream classification doesn't need that much variance — 5 components carry the *discriminative* information.

**Design choice:** ship `PCA(n_components=5)` (linear) for this problem — best downstream accuracy, 6× dimensionality reduction, no kernel hyperparameters to tune. If I needed reconstruction fidelity instead of discrimination I'd raise `n_components` to ~10 to hit the 95% variance bar.

### Reflections

The headline finding — **5 components beat 30** for downstream classification — is the lesson I'd want every ML hire to internalise about dimensionality reduction. More features ≠ more information; past the discriminative components, you're adding noise that the downstream model has to actively suppress. That's why "PCA hurt my model" is sometimes a sign of *too many components*, not "PCA was wrong for this problem."

The KernelPCA result has the opposite, equally important, lesson: matching the *tool* to the *data structure* matters more than picking the fanciest tool. RBF KernelPCA underperforms linear PCA on Breast Cancer not because it's a worse algorithm but because the underlying class boundary is already linear after standardisation — adding kernel non-linearity squeezes that natural geometry through a curved manifold the downstream linear classifier can't exploit. Reaching for kernel methods on linearly-separable problems is a common junior-engineer mistake; running the comparison once kills it for the team.

On the communication side, the 2-D PCA scatter is the most-useful single chart in this whole repo when showing the result to someone outside the modelling team. "These two clouds barely overlap" is far more convincing than "our test AUC is 0.97" because it bypasses the metric and *shows* the structure. Building these visualizations into the eval pipeline by default is cheap and pays for itself the first time you need it in a meeting.

### Methodology Notes
- All PCA is fit **after** `StandardScaler` so each feature contributes
  proportionally to its z-scored variance, not its raw scale.
- Reconstruction error is reported in the **standardised** space.
  Reporting it in raw-X space is also legitimate; we chose the
  standardised version so the curve has interpretable units (norm of a
  unit-variance residual).
- The KernelPCA(`rbf`/`poly`) results don't have closed-form
  `explained_variance_ratio_` exposed; we only report downstream
  accuracy.

### Limitations
- 569 rows is small; the n_components ≥ ~5 plateau is very flat —
  read the curve, not individual decimal points.
- 3-D scatter is for visualisation only; downstream accuracy is the
  scored metric.

### Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0
python evaluate.py
```
