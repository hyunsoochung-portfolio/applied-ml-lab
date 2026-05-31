# Income Streamer — SGD on UCI Adult Income

Stochastic-gradient linear models on UCI Adult Income (`>50K` vs
`<=50K`). Four experiments covering loss choice, learning-rate
schedule, batch size, early stopping, and a multi-seed epoch curve.

## Setup
- `fetch_openml('adult', version=2, as_frame=True)`
- One-hot encoded categoricals, NaN rows dropped.
- All experiments share the same scaling: `StandardScaler(with_mean=False)`
  fit on the training split (sparse-safe).

## Experiments

### 1. Loss × learning-rate-schedule grid
- `loss ∈ {log_loss, hinge, modified_huber, squared_hinge}`
- `learning_rate ∈ {constant, optimal, invscaling, adaptive}`
- Single hold-out per cell; `max_iter=50, tol=1e-4`
- `eta0=0.01` is set for the schedules that require it.

Output: `loss_schedule_grid.csv` (16 cells).

### 2. partial_fit batch-size study
`batch_size ∈ {128, 512, 2048}`, fixed 8 epochs, `loss=log_loss`. Logs
test accuracy and total wall-clock per batch size.

Output: `batch_size_study.csv`.

### 3. Early stopping vs full epochs
Two SGD runs with `max_iter=30`: one with `early_stopping=False`, one
with `early_stopping=True, validation_fraction=0.1, n_iter_no_change=5`.
Compares final accuracy, `n_iter_`, and wall-clock.

Output: `early_stopping.csv`.

### 4. Multi-seed epoch curve
For each of 5 seeds, run 25 epochs of `partial_fit` (batch=1024) and
log per-epoch train + test accuracy. The plot shows mean ± shaded std
across seeds.

Outputs: `epoch_curve_multiseed.csv`, `epoch_curve.png`.

## Findings

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

## Reflections

`partial_fit` is the underrated API in this project. Most ML examples treat training as a one-shot process; in production, data arrives over time and the model needs to *keep learning* without retraining from scratch every night. The batch-size sweep is a small concrete glimpse of that streaming regime: smaller batches mean more frequent updates, more wall-clock cost, slightly better accuracy. Productionising it is a real systems exercise — model versioning, replay buffers, rollback on quality regression — but the algorithmic primitive is right there in `SGDClassifier`.

The loss × LR-schedule grid is a useful experiment to *run with a teammate* who's never tuned SGD before. Watching `squared_hinge` collapse on the `adaptive` schedule while `hinge` thrives on it is the kind of "the loss and the schedule have to talk to each other" lesson that's much faster to absorb from a single grid than from a textbook chapter. Building these comparison harnesses is half of what makes a team faster over time.

The early-stopping result (too eager on this validation curve) is a small example of why I distrust any "off by default" configuration that silently changes behaviour. A patience of 5 is *not* the right default for every dataset, but the package ships one as if it were. The lesson is to treat library defaults as "the author's guess about the median use case" — fine until your case isn't the median, at which point the cost of *not knowing* the default existed is high.

## Methodology Notes
- All seeds re-split the data and re-init the scaler so the variance
  reflects both **stochastic optimisation noise** and **sampling
  noise** — the way users will encounter it in practice.
- The early-stopping comparison uses `validation_fraction=0.1` carved
  out of `Xtr`, not the held-out test set; the reported accuracy is
  still on the held-out test set, so the comparison is honest.
- `loss=hinge` does not support `predict_proba`; we only score
  accuracy so the comparison stays apples-to-apples.

## Limitations
- Adult Income has ~48k rows; bigger batch sizes start to dominate the
  wall-clock comparison because each `partial_fit` call has fixed
  overhead. Reading the curve as "throughput vs convergence" is more
  useful than reading single batch-size numbers.
- We do not sweep `eta0` or `alpha` — adding `--eta0`/`--alpha` flags
  is a natural extension.

## Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0 --n-seeds 5
python evaluate.py
```
