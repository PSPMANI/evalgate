"""Drift monitor: is a new batch of customers still like the data the model learned from?

    python monitor.py batch.csv                  # exit 1 if any feature has shifted significantly
    python monitor.py --make-shifted-demo out.csv  # write a deliberately shifted demo batch

Compares each feature's distribution in the batch against data_profile.json (written by
train.py from the training split) using the population stability index (PSI):
below 0.10 stable, 0.10 to 0.25 moderate, above 0.25 significant (blocks).
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

from evalgate.checks import drift_checks, load_config, to_markdown
from evalgate.metrics import drift

HERE = pathlib.Path(__file__).parent


def make_shifted_demo(out: pathlib.Path) -> None:
    """A plausible real-world shift: a price rise, and the mix tilting to monthly plans."""
    df = pd.read_csv(HERE / "data" / "telco_churn.csv")
    rng = np.random.default_rng(7)
    df["MonthlyCharges"] = (df["MonthlyCharges"] * 1.25).round(2)
    flip = (df["Contract"] != "Month-to-month") & (rng.random(len(df)) < 0.6)
    df.loc[flip, "Contract"] = "Month-to-month"
    df.to_csv(out, index=False)
    print(f"wrote {out}: MonthlyCharges +25%, {int(flip.sum())} customers moved to monthly contracts")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", nargs="?", help="CSV with the model's input columns")
    ap.add_argument("--make-shifted-demo", metavar="OUT")
    args = ap.parse_args()
    if args.make_shifted_demo:
        make_shifted_demo(pathlib.Path(args.make_shifted_demo))
        return 0
    if not args.batch:
        ap.error("give a batch CSV")

    cfg = load_config()
    prof = json.loads((HERE / "data_profile.json").read_text(encoding="utf-8"))
    batch = pd.read_csv(args.batch)
    if "TotalCharges" in batch:
        batch["TotalCharges"] = pd.to_numeric(batch["TotalCharges"], errors="coerce")
    scores = drift(prof, batch.drop(columns=["Churn", "customerID"], errors="ignore"))
    checks = drift_checks(scores, cfg)
    for c in checks:
        level = "FAIL" if not c.passed else ("warn" if float(c.value) > cfg["drift"]["psi_warn"] else "ok")
        print(f"[{level:>4}] {c.name:<24} {c.value}")
    (HERE / "drift_report.md").write_text(to_markdown(checks, "EvalGate drift monitor"), encoding="utf-8")
    shifted = [c.name for c in checks if not c.passed]
    if shifted:
        print(f"\nDRIFT BLOCKED: {', '.join(shifted)} shifted significantly; retrain or investigate.")
        return 1
    print("\nNo significant drift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
