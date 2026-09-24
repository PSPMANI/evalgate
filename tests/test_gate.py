"""Gate logic tests: the thing that blocks deployments must itself be tested."""
import copy

import pytest

from evalgate.checks import load_config, run_gate, to_markdown

CFG = load_config()


def slices(**overrides):
    base = {"gender": {"Female": {"n": 600, "base_rate": 0.27, "auc": 0.83},
                       "Male": {"n": 700, "base_rate": 0.26, "auc": 0.84}},
            "SeniorCitizen": {"0": {"n": 1100, "base_rate": 0.24, "auc": 0.84},
                              "1": {"n": 250, "base_rate": 0.41, "auc": 0.79}},
            "Contract": {"One year": {"n": 290, "base_rate": 0.14, "auc": 0.70},
                         "Tiny": {"n": 12, "base_rate": 0.5, "auc": 0.40}}}
    base.update(overrides)
    return base


GOOD = {"roc_auc_test": 0.84, "accuracy_test": 0.80, "roc_auc_cv_mean": 0.845, "ece": 0.03,
        "brier": 0.14, "slices": slices(), "drift_test_vs_train": {"tenure": 0.01, "Contract": 0.002}}


def failed(metrics, comparison=None):
    return {c.name for c in run_gate(metrics, CFG, comparison) if not c.passed}


def test_policy_file_has_every_section():
    assert {"thresholds", "calibration", "slices", "champion", "drift"} <= set(CFG)


def test_good_model_passes():
    assert failed(GOOD) == set()


@pytest.mark.parametrize("key,value,expected", [
    ("roc_auc_test", 0.70, {"roc_auc_test", "cv_test_gap"}),
    ("accuracy_test", 0.70, {"accuracy_test"}),
    ("roc_auc_cv_mean", 0.90, {"cv_test_gap"}),  # CV far above test: leakage smell
    ("ece", 0.09, {"ece"}),
    ("brier", 0.20, {"brier"}),
])
def test_each_threshold_blocks(key, value, expected):
    assert failed({**GOOD, key: value}) == expected


def test_slice_floor_blocks_a_collapsed_segment():
    m = copy.deepcopy(GOOD)
    m["slices"]["Contract"]["One year"]["auc"] = 0.60
    assert failed(m) == {"Contract=One year"}


def test_small_slices_are_reported_but_not_gated():
    checks = run_gate(GOOD, CFG)
    tiny = next(c for c in checks if c.name == "Contract=Tiny")
    assert tiny.skipped and tiny.passed


def test_fairness_gap_blocks():
    m = copy.deepcopy(GOOD)
    m["slices"]["SeniorCitizen"]["1"]["auc"] = 0.74
    assert failed(m) == {"SeniorCitizen AUC gap"}


def test_champion_non_inferiority():
    assert failed(GOOD, {"diff": -0.004, "ci": [-0.009, 0.001]}) == set()
    assert failed(GOOD, {"diff": -0.010, "ci": [-0.017, -0.002]}) == {"non-inferior to deployed model"}


def test_no_champion_is_skipped_not_failed():
    c = next(c for c in run_gate(GOOD, CFG) if c.group == "champion")
    assert c.skipped and c.passed


def test_train_test_drift_blocks():
    assert failed({**GOOD, "drift_test_vs_train": {"tenure": 0.4}}) == {"PSI tenure"}


def test_markdown_report():
    md = to_markdown(run_gate({**GOOD, "ece": 0.2}, CFG))
    assert md.startswith("## EvalGate: BLOCKED") and "**FAIL**" in md
