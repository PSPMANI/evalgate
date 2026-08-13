"""The quality gate: fail the pipeline if the model is not good enough to ship.

    python gate.py

Reads metrics.json (produced by train.py) and enforces hard thresholds.
Exit code 0 = model may be deployed. Exit code 1 = deployment is BLOCKED.
CI runs this as its own step, so a weak model turns the pipeline red.
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent

THRESHOLDS = {
    "roc_auc_test": 0.82,
    "accuracy_test": 0.75,
    "roc_auc_cv_mean": 0.82,
}

# CV must agree with the test score to within this margin, or something is
# wrong with the split (a drift/leakage smell), and we block.
MAX_CV_TEST_GAP = 0.03


def main():
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
    failures = []

    for key, minimum in THRESHOLDS.items():
        value = metrics.get(key, 0.0)
        status = "PASS" if value >= minimum else "FAIL"
        print(f"[{status}] {key}: {value} (required >= {minimum})")
        if value < minimum:
            failures.append(key)

    gap = abs(metrics.get("roc_auc_test", 0) - metrics.get("roc_auc_cv_mean", 0))
    status = "PASS" if gap <= MAX_CV_TEST_GAP else "FAIL"
    print(f"[{status}] cv-vs-test gap: {gap:.4f} (required <= {MAX_CV_TEST_GAP})")
    if gap > MAX_CV_TEST_GAP:
        failures.append("cv_test_gap")

    if failures:
        print(f"\nGATE BLOCKED: {', '.join(failures)} - this model does not ship.")
        sys.exit(1)
    print("\nGATE PASSED: model quality is acceptable, deployment may proceed.")


if __name__ == "__main__":
    main()
