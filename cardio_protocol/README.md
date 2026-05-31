# Cardio Protocol — Data Splits & Cross-Validation on UCI Cleveland Heart Disease

A small R&D study of how data-splitting choices affect what your
"validation accuracy" actually means. Four experiments, all logged to
`artifacts/` as CSV + PNG.

## Setup
- Same dataset as `cardio_screener` (Cleveland Heart Disease).
- Reference estimator: `StandardScaler -> LogisticRegression(max_iter=2000)`
  built by `build_classifier()` in `model.py`.

## Experiments

### 1. Splitter variance across 10 seeds
For each seed we record the score from:
- `train_test_split` (random hold-out)
- `train_test_split(stratify=y)` (stratified hold-out)
- `KFold(5)`
- `StratifiedKFold(5)`
- `RepeatedStratifiedKFold(5, n_repeats=2)`

Output: `splitter_variance.csv` (one row per fold per seed),
`splitter_variance.png` (boxplot per splitter).

### 2. Fold-count sensitivity
Sweeps `K ∈ {3, 5, 10, 20}` of `StratifiedKFold`, repeating across all
seeds. Logs within-CV std (fold-to-fold) and across-seed std (run-to-run).

Output: `fold_count.csv`.

### 3. Nested CV
Outer `StratifiedKFold(5)` for model selection, inner `StratifiedKFold(3)`
for `LogisticRegression(C ∈ {0.01, 0.1, 1, 10})`. Records the chosen `C`
per outer fold along with the inner-best and outer-test scores.

Output: `nested_cv.csv`.

### 4. Leakage quantification
Two pipelines for the *same* model & splits:
- **leaky**: `StandardScaler` fit on the full `X` before CV.
- **clean**: scaler inside the sklearn `Pipeline`, fit per fold.

Output: `leakage.csv` with an `inflation = leaky - clean` column per seed.

## Findings

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

## Reflections

The leakage demo is the heart of this project for me. On 303 rows the measured inflation was tiny (≤0.003), but the *structural* mistake is identical to the one that ships +5 pp "improvements" on larger feature spaces and then quietly degrades in production. Most data-science bugs that survive into prod look like that — small effects on the metric you check, large effects on the world. Putting the scaler inside the `Pipeline` is a one-line fix that closes that whole class of bug forever; the harder thing is explaining to a teammate *why* the "obviously correct" `scaler.fit_transform(X)` outside CV is actually wrong.

Cross-validation is also where "what does this metric mean" conversations come up. Hold-out looks more stable than 5-fold CV across seeds, but only because it averages nothing — each draw is a single noisy estimate. Saying it plainly — "the smaller-looking number is the more honest one" — is the kind of explanation that buys trust over time.

On the engineering side, nested CV is honest model-selection but it's also 15× the compute of a single fit. There's a real product judgement under that: when does the conservatism of nested CV pay for itself, and when is "naive 5-fold + a willingness to re-train on fresh data" cheaper and just as safe? The right answer depends on how fast your data drifts, not on what's nominally "more correct."

## Methodology Notes
- We never call `.fit_transform` on the full dataset for any reported
  metric except the explicit leakage demo — that's the whole point.
- `StratifiedKFold` is the default everywhere except (a) the
  non-stratified hold-out, (b) plain `KFold` used as a comparison
  baseline; both included so the variance gap is visible.
- "Splitter variance" intentionally mixes hold-out (1 score per seed)
  with k-fold (5 scores per seed); the boxplot per splitter is the
  right object to look at.

## Limitations
- ~297 rows after cleaning is small — variance estimates themselves
  carry noise. The qualitative pattern (`KFold` widest, `RSKF` tightest)
  is stable; absolute numbers will jiggle across machines.
- Nested CV uses only 4 candidate `C` values to keep wall-clock low;
  expand `grid` in `nested_cv()` for a finer sweep.

## Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0 --n-seeds 10
python evaluate.py
```
