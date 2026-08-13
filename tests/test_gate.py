"""Gate logic tests: the thing that blocks deployments must itself be tested."""
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent.parent


def run_gate_with(metrics, tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (tmp_path / "gate.py").write_text(
        (HERE / "gate.py").read_text(encoding="utf-8"), encoding="utf-8")
    return subprocess.run([sys.executable, str(tmp_path / "gate.py")],
                          capture_output=True, text=True)


GOOD = {"roc_auc_test": 0.84, "accuracy_test": 0.80, "roc_auc_cv_mean": 0.845}
BAD = {"roc_auc_test": 0.70, "accuracy_test": 0.80, "roc_auc_cv_mean": 0.71}
DRIFTED = {"roc_auc_test": 0.90, "accuracy_test": 0.80, "roc_auc_cv_mean": 0.83}


def test_gate_passes_good_model(tmp_path):
    r = run_gate_with(GOOD, tmp_path)
    assert r.returncode == 0
    assert "GATE PASSED" in r.stdout


def test_gate_blocks_weak_model(tmp_path):
    r = run_gate_with(BAD, tmp_path)
    assert r.returncode == 1
    assert "GATE BLOCKED" in r.stdout


def test_gate_blocks_cv_test_disagreement(tmp_path):
    r = run_gate_with(DRIFTED, tmp_path)
    assert r.returncode == 1
    assert "cv_test_gap" in r.stdout
