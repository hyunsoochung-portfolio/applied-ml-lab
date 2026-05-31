# Churn Alert — Logistic Regression on Telco Customer Churn

A logistic-regression R&D study: solver / C / class-weight sweeps,
calibration + threshold curves, and a synthetic 3-class side experiment
for OvR vs multinomial.

## Setup
- IBM mirror CSV: <https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv>
- Falls back to `fetch_openml('telco-customer-churn', as_frame=True)`
  if the mirror is unreachable.
- ~7 000 rows, one-hot encoded -> mid-30s features, ~26% positive class.

## Experiments

### 1. Solver comparison
`lbfgs` / `saga` / `liblinear` on the **same** problem with 5-fold
`StratifiedKFold` ROC-AUC. `liblinear` only does OvR for multiclass but
binary is fine. Warning suppression is on so the table doesn't get
swamped on slow solvers.

Output: `solver_comparison.csv`.

### 2. C regularization sweep
`C ∈ np.logspace(-3, 2, 11)`, 5-fold CV ROC-AUC.
Outputs: `c_sweep.csv`, `c_sweep.png` (errorbar plot, log x).

### 3. class_weight ablation
`class_weight ∈ {None, 'balanced'}` on a stratified 75/25 hold-out.
Logs minority recall alongside accuracy and AUC — the right metric for
this trade-off.

Output: `class_weight_ablation.csv`.

### 4. Calibration curve + Brier score
Calibration on 10 quantile bins, plus the Brier score for the best CV
C. Logistic regression with a scaler is usually well-calibrated, so the
curve sits near the diagonal — useful sanity check.

Output: `calibration.png`.

### 5. ROC + Precision-Recall curves
Both curves rendered on the same held-out split.

Output: `roc_pr_curves.png`.

### 6. OvR vs multinomial (3-class)
Synthetic 3-class target = `tenure` bucketed at quantiles 1/3 and 2/3.
Compares `OneVsRestClassifier` against multinomial `LogisticRegression`,
checks that softmax rows sum to 1 while OvR does not.

Output: `multiclass_ovr_vs_softmax.csv`.

## Findings

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

## Reflections

The class-weight ablation is the most product-relevant finding in the module: `balanced` trades 5 pp accuracy for +26 pp minority recall while leaving ROC-AUC essentially unchanged. The model didn't actually get "better" or "worse" — we moved the threshold. That distinction matters because the right threshold isn't a modelling decision; it's a *product* decision about asymmetric costs. For churn, missing a churner costs much more than a false alarm, so balanced wins. For an irreversible action (e.g. account suspension) the calculus flips. Getting the engineering team and the product team to talk about *the cost ratio* up front is half the value of building this kind of model.

The solver comparison (all three identical to four decimals) is a less glamorous insight that pays off in code reviews: when three optimizers reach the same minimum on the same convex loss, that's the loss telling you it's well-conditioned. The choice between lbfgs / saga / liblinear is an *operational* decision (data scale, regularization type, parallelism), not a modelling one. Recognising "this is operational, not modelling" early stops a lot of bikeshed.

On the systems side, the calibration result and Brier score matter for any downstream system that *uses* the predicted probability as input — recommendation re-ranking, threshold-based routing, expected-value math. A well-calibrated LR with `balanced` weights is the kind of unfashionable baseline that quietly wins production deployments because the downstream consumers can trust the numbers.

## Methodology Notes
- All preprocessing (scaler, one-hot already in `load_binary`) is
  inside the sklearn `Pipeline` so each CV fold gets its own scaler.
- The 3-class target intentionally drops `tenure` and `Churn` from
  features to avoid trivial leakage.
- Calibration is reported with `strategy="quantile"` so bins have
  comparable counts even when the score distribution is skewed (which
  it always is on churn problems).

## Limitations
- `saga` may need more iterations on this problem; we cap at 4000 to
  keep wall-clock reasonable. If you see convergence warnings, raise
  it in `build_binary(solver='saga')`.
- The 3-class target is constructed for educational comparison, not a
  real product target.

## Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0
python evaluate.py
```
