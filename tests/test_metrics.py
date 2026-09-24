"""Metric tests: every statistic the gate trusts is checked on hand-computable inputs."""
import math

import numpy as np
import pandas as pd
import pytest

from evalgate.metrics import drift, ece, paired_auc_diff, profile, psi, slice_metrics


def test_ece_is_zero_for_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    assert ece(y, p) < 0.01


def test_ece_detects_overconfidence():
    p = np.full(1000, 0.9)
    y = np.r_[np.ones(500), np.zeros(500)]
    assert ece(y, p) == pytest.approx(0.4)


def test_psi_known_value():
    expected = 0.4 * math.log(0.9 / 0.5) + (-0.4) * math.log(0.1 / 0.5)
    assert psi([0.5, 0.5], [0.9, 0.1]) == pytest.approx(expected)
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0.0


def test_drift_is_near_zero_on_the_same_data_and_flags_a_shift():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"charges": rng.normal(70, 20, 5000),
                        "plan": rng.choice(["monthly", "yearly"], 5000, p=[0.5, 0.5])})
    prof = profile(ref)
    assert max(drift(prof, ref).values()) < 0.01
    shifted = ref.assign(charges=ref["charges"] * 1.25,
                         plan=rng.choice(["monthly", "yearly"], 5000, p=[0.9, 0.1]))
    scores = drift(prof, shifted)
    assert scores["charges"] > 0.25 and scores["plan"] > 0.25


def test_drift_handles_unseen_categories():
    prof = profile(pd.DataFrame({"plan": ["a", "b"] * 50}))
    assert drift(prof, pd.DataFrame({"plan": ["c"] * 100}))["plan"] > 1.0


def test_paired_diff_is_zero_for_identical_models():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 500)
    p = rng.uniform(0, 1, 500)
    assert paired_auc_diff(y, p, p) == {"diff": 0.0, "ci": [0.0, 0.0]}


def test_paired_diff_sign_follows_the_better_model():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 2000)
    good = y + rng.normal(0, 0.6, y.size)
    bad = y + rng.normal(0, 1.5, y.size)
    r = paired_auc_diff(y, bad, good)
    assert r["diff"] < 0 and r["ci"][1] < 0


def test_slice_metrics():
    frame = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 4, "y": [0, 1, 0, 1, 0, 0, 0, 0],
                          "p": [0.1, 0.9, 0.2, 0.8, 0.1, 0.2, 0.3, 0.4]})
    s = slice_metrics(frame, ["g"])
    assert s["g"]["a"] == {"n": 4, "base_rate": 0.5, "auc": 1.0}
    assert s["g"]["b"]["auc"] is None  # one class only: AUC undefined, reported not faked
