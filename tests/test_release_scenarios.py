"""End-to-end release scenarios on the real data: the gate must stop what v1 would ship."""
import pathlib
import sys

import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from evalgate.checks import champion_check, load_config  # noqa: E402
from evalgate.metrics import drift, paired_auc_diff, profile  # noqa: E402
from monitor import make_shifted_demo  # noqa: E402
from train import SEED, build_pipeline, load_data  # noqa: E402

CFG = load_config()


@pytest.fixture(scope="module")
def split():
    X, y = load_data()
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def test_subtle_regression_passes_the_floor_but_not_the_champion_check(split):
    X_tr, X_te, y_tr, y_te = split
    champion = build_pipeline(X_tr).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    weaker = build_pipeline(X_tr, n_estimators=20, max_depth=2).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    # The absolute floor alone (all v1 had) would let this model ship...
    assert roc_auc_score(y_te, weaker) >= CFG["thresholds"]["roc_auc_test"]
    # ...but it is measurably worse than the deployed model, so v2 blocks it.
    comparison = paired_auc_diff(y_te.values, weaker, champion, n_boot=300)
    assert not champion_check(comparison, CFG)[0].passed


def test_drift_monitor_flags_the_shifted_demo_batch(split, tmp_path):
    X_tr, X_te, _, _ = split
    prof = profile(X_tr)
    assert max(drift(prof, X_te).values()) < CFG["drift"]["psi_warn"]
    out = tmp_path / "shifted.csv"
    make_shifted_demo(out)
    batch = pd.read_csv(out).drop(columns=["Churn", "customerID"])
    batch["TotalCharges"] = pd.to_numeric(batch["TotalCharges"], errors="coerce")
    scores = drift(prof, batch)
    blocked = {k for k, v in scores.items() if v > CFG["drift"]["psi_block"]}
    assert blocked == {"MonthlyCharges", "Contract"}
