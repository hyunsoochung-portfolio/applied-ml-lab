"""Tiny shared helpers used across the 13 modules.

Kept intentionally small — every module is otherwise self-contained.
This module also exposes a handful of R&D helpers used by the
``experiments/`` blocks: multi-seed metric aggregation, CSV writers,
and a small Pareto-front utility.
"""
from __future__ import annotations

import csv as _csv
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Sequence, Tuple

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed Python ``random`` and NumPy for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create ``path`` (and parents) if missing; return as :class:`Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def timer(label: str = "block") -> Iterator[None]:
    """Context manager that prints elapsed wall-clock seconds."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[timer] {label}: {dt:.3f}s")


def save_plot(fig, out_path: str | os.PathLike, *, dpi: int = 130) -> Path:
    """Save a matplotlib ``Figure`` to ``out_path`` and close it.

    Creates the parent directory if needed. Returns the resolved path.
    """
    out = Path(out_path)
    ensure_dir(out.parent)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    # close to release memory in long scripts
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    print(f"[plot] saved -> {out}")
    return out


def data_cache_dir(module_file: str) -> Path:
    """Return ``<module_dir>/data`` and create it. ``module_file`` is ``__file__``."""
    return ensure_dir(Path(module_file).resolve().parent / "data")


# ---------------------------------------------------------------------------
# R&D helpers — used by every module's ``experiments/`` block
# ---------------------------------------------------------------------------


def artifacts_dir(module_file: str, sub: str | None = None) -> Path:
    """Return ``<module>/artifacts[/sub]`` (created). Used by experiments."""
    base = Path(module_file).resolve().parent / "artifacts"
    if sub:
        base = base / sub
    return ensure_dir(base)


def multi_seed_metric(
    func: Callable[..., float],
    seeds: Sequence[int],
    **kwargs,
) -> Tuple[float, float, List[float]]:
    """Run ``func(seed=s, **kwargs)`` across ``seeds`` and return (mean, std, raw).

    ``func`` must accept ``seed`` as a keyword and return a scalar.
    """
    raw: List[float] = []
    for s in seeds:
        v = float(func(seed=int(s), **kwargs))
        raw.append(v)
    arr = np.asarray(raw, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0)), raw


def save_csv(rows: Iterable[dict], path: str | os.PathLike) -> Path:
    """Write a list of dict rows to CSV (header = union of all keys)."""
    rows = list(rows)
    out = Path(path)
    ensure_dir(out.parent)
    if not rows:
        out.write_text("")
        print(f"[csv] wrote (empty) -> {out}")
        return out
    fieldnames: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames, quoting=_csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[csv] wrote {len(rows)} rows -> {out}")
    return out


def pareto_front(
    points: Sequence[Tuple[float, float]],
    *,
    minimize_x: bool = True,
    maximize_y: bool = True,
) -> List[int]:
    """Return indices of non-dominated points in ``points`` (list of (x,y)).

    Default convention: minimise x (e.g. wall-clock seconds), maximise y
    (e.g. CV score). A point ``p`` dominates ``q`` if it is no worse on
    both axes and strictly better on at least one.
    """
    n = len(points)
    keep: List[int] = []
    for i in range(n):
        xi, yi = points[i]
        dominated = False
        for j in range(n):
            if i == j:
                continue
            xj, yj = points[j]
            better_x = (xj <= xi) if minimize_x else (xj >= xi)
            better_y = (yj >= yi) if maximize_y else (yj <= yi)
            strictly = (xj != xi) or (yj != yi)
            if better_x and better_y and strictly:
                # j is at least as good on both, strictly better somewhere
                strictly_better_x = (xj < xi) if minimize_x else (xj > xi)
                strictly_better_y = (yj > yi) if maximize_y else (yj < yi)
                if strictly_better_x or strictly_better_y:
                    dominated = True
                    break
        if not dominated:
            keep.append(i)
    return keep
