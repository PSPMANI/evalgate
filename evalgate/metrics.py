"""Model-quality statistics the gate relies on.

Everything here is a plain function over numpy arrays, so each one is unit-tested on
hand-checkable inputs before the gate is allowed to trust it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: bin by predicted probability, then average
    |mean prediction - observed rate| weighted by bin size."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def reliability_curve(y, p, n_bins: int = 10) -> list[dict]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            out.append({"bin": b, "n": int(mask.sum()), "mean_pred": round(float(p[mask].mean()), 4),
                        "observed": round(float(y[mask].mean()), 4)})
    return out


def auc_ci(y, p, n_boot: int = 1000, seed: int = 0) -> list[float]:
    """Percentile bootstrap 95% CI for ROC-AUC."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            stats.append(roc_auc_score(y[i], p[i]))
    return [round(float(np.percentile(stats, 2.5)), 4), round(float(np.percentile(stats, 97.5)), 4)]


def paired_auc_diff(y, p_new, p_old, n_boot: int = 1000, seed: int = 0) -> dict:
    """Challenger minus champion AUC on the SAME rows, with a paired bootstrap CI.

    Pairing matters: both models are scored on each resample, so row-level noise
    cancels and the interval measures the models' difference, not the test set's luck.
    """
    y, p_new, p_old = np.asarray(y), np.asarray(p_new), np.asarray(p_old)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            diffs.append(roc_auc_score(y[i], p_new[i]) - roc_auc_score(y[i], p_old[i]))
    point = roc_auc_score(y, p_new) - roc_auc_score(y, p_old)
    return {"diff": round(float(point), 4),
            "ci": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]}


def slice_metrics(frame: pd.DataFrame, columns: list[str], y_col: str = "y", p_col: str = "p") -> dict:
    """AUC, size and base rate for every value of every slice column."""
    out = {}
    for col in columns:
        out[col] = {}
        for value, g in frame.groupby(col, observed=True):
            auc = roc_auc_score(g[y_col], g[p_col]) if g[y_col].nunique() > 1 else None
            out[col][str(value)] = {"n": int(len(g)), "base_rate": round(float(g[y_col].mean()), 4),
                                    "auc": None if auc is None else round(float(auc), 4)}
    return out


# ---- data drift: population stability index -------------------------------------

def _bins(reference: pd.Series, n_bins: int = 10) -> list[float]:
    qs = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, n_bins + 1)))
    return [float(x) for x in qs[1:-1]]  # interior edges; the outer bins are open-ended


def profile(frame: pd.DataFrame) -> dict:
    """Reference distribution of every feature, saved at training time."""
    prof = {}
    for col in frame.columns:
        s = frame[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            edges = _bins(s)
            counts = np.bincount(np.searchsorted(edges, s.dropna(), side="right"),
                                 minlength=len(edges) + 1)
            prof[col] = {"kind": "numeric", "edges": edges,
                         "share": [round(float(c / counts.sum()), 6) for c in counts]}
        else:
            share = s.astype(str).value_counts(normalize=True)
            prof[col] = {"kind": "categorical",
                         "share": {str(k): round(float(v), 6) for k, v in share.items()}}
    return prof


def psi(expected: list[float], actual: list[float], eps: float = 1e-4) -> float:
    """Population stability index: sum((a - e) * ln(a / e)). Rule of thumb:
    < 0.10 stable, 0.10 - 0.25 moderate shift, > 0.25 significant shift."""
    e = np.clip(np.asarray(expected, float), eps, None)
    a = np.clip(np.asarray(actual, float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift(prof: dict, batch: pd.DataFrame) -> dict:
    """PSI of every profiled feature between the training reference and a new batch."""
    out = {}
    for col, ref in prof.items():
        if col not in batch.columns:
            continue
        s = batch[col]
        if ref["kind"] == "numeric":
            counts = np.bincount(np.searchsorted(ref["edges"], pd.to_numeric(s, errors="coerce").dropna(),
                                                 side="right"), minlength=len(ref["edges"]) + 1)
            actual = counts / max(counts.sum(), 1)
            out[col] = round(psi(ref["share"], actual), 4)
        else:
            share = s.astype(str).value_counts(normalize=True)
            cats = sorted(set(ref["share"]) | set(share.index))
            out[col] = round(psi([ref["share"].get(c, 0.0) for c in cats],
                                 [float(share.get(c, 0.0)) for c in cats]), 4)
    return out
