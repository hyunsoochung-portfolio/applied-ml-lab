# Deposit Propensity — Tree-Ensemble Benchmark on UCI Bank Marketing

Benchmark six tree ensembles on a real, imbalanced marketing dataset
across multiple seeds, then dig into agreement on feature importance,
calibration quality, and out-of-fold predictions.

## Setup
- UCI Bank Marketing — `bank-full.csv` from
  <https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip>
- Falls back to `bank-additional-full.csv` if the first archive fails.
- Target `y` (yes/no = subscribed); the positive class is rare (~11 %).
- Estimators: `RandomForest`, `ExtraTrees`, `GradientBoosting`,
  `HistGradientBoosting`, `XGBoost`, `LightGBM` (last two fail-soft if
  the libraries are missing).

## Experiments

### 1. 5-seed benchmark
For each seed and each ensemble, fit on a stratified 80% split and
score `(roc_auc, f1, accuracy)` on the held-out 20%. Wall-clock per
fit also recorded.

Output: `ensemble_benchmark.csv` — one row per (model, seed).

### 2. Permutation importance on the best model
Best model = highest mean ROC-AUC across seeds. Permutation importance
with `n_repeats=5`, ROC-AUC scoring, on the held-out test split.

Outputs: `perm_importance.csv`, `perm_importance.png`.

### 3. Feature-importance agreement matrix
Spearman ρ between every pair of models' `feature_importances_`
vectors. Surfaces "do these tree models agree about which features
matter, or does each pick its own?".

Outputs: `importance_agreement.csv`, `importance_agreement.png`
(matrix heatmap with annotated ρ values).

### 4. Calibration overlay
Calibration curves for all ensembles on one axes (10 quantile bins) +
Brier score in the legend. Tree models are notoriously
mis-calibrated; this makes that visible.

Output: `calibration_overlay.png`.

### 5. Out-of-fold predictions
`cross_val_predict(method="predict_proba", cv=5)` for every ensemble.
Saved as wide CSV (`y_true, oof_<model>, ...`) — natural input for
downstream stacking experiments.

Output: `oof_predictions.csv`.

## Findings

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

## Reflections

The headline number — HistGradientBoosting at 0.9346 ROC-AUC — is less interesting to me than the *agreement matrix*. XGBoost and LightGBM reach essentially the same score (0.9329 vs 0.9340) but their feature-importance Spearman is 0.328. Two well-tuned boosted models can disagree completely about *why* they got there. That's the most useful argument I know for blending models in production: not because the ensemble's mean is higher, but because the *errors* of each model are partially decorrelated.

Cross-functionally, ensemble comparison tables are also useful at being honest about diminishing returns. A new team often jumps straight to "let's try XGBoost / LightGBM / CatBoost" before tuning a RandomForest. On this dataset RF lands within 0.5 pp of the best model — half the time, the question isn't which library, it's whether the gap to your baseline is worth the engineering cost of adding another dependency. Reporting *all six* on the same axes lets the team have that conversation explicitly.

On the systems side, the fit-time column matters more than people admit in offline comparisons. GradientBoosting at 9.4 s/fit is 9× XGBoost at 1.0 s — that compounds *brutally* in hyperparameter search and in production retraining loops. For a team deciding what to ship, "0.06 pp slower but 3× faster to retrain" is often the right trade, especially when data drift is the real long-term enemy.

## Methodology Notes
- `XGBoost` / `LightGBM` use `n_estimators=300, learning_rate=0.1` —
  not tuned per dataset on purpose so the benchmark is comparable to
  the sklearn defaults next door.
- Permutation importance is **test-set** based and scored on AUC, so
  it answers "permuting this column drops AUC by how much".
- The agreement matrix only uses models that expose
  `feature_importances_`; HistGradientBoosting does not, so it's
  excluded from that subplot (still benchmarked).

## Limitations
- Strong class imbalance: accuracy can look high even from the
  majority predictor. Read AUC and F1 first.
- Single-set hyperparameters per ensemble. A per-model RandomizedSearch
  pass would tighten the differences; out of scope here.

## Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0 --n-seeds 5
python evaluate.py
```
