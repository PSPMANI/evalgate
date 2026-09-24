"""The release gate: block the pipeline unless the model is fit to ship.

    python gate.py                        # compare against the deployed champion
    python gate.py --champion none        # first deployment / offline run

Reads metrics.json and predictions.json (from train.py) and the policy in gate.toml,
then checks quality, calibration, every customer slice, fairness gaps, non-inferiority
to the model currently deployed, and train/test feature drift.

Exit code 0 = the model may be deployed. Exit code 1 = deployment is BLOCKED.
Writes gate_report.json and a Markdown table (also to $GITHUB_STEP_SUMMARY in CI).
"""
import argparse
import json
import os
import pathlib
import sys
import urllib.request

from evalgate.checks import load_config, run_gate, to_markdown
from evalgate.metrics import paired_auc_diff

HERE = pathlib.Path(__file__).parent
CHAMPION_URL = "https://pspmani.github.io/evalgate/champion.json"


def load_champion(source: str):
    if source == "none":
        return None
    try:
        if source.startswith("http"):
            with urllib.request.urlopen(source, timeout=10) as r:
                return json.load(r)
        return json.loads(pathlib.Path(source).read_text(encoding="utf-8"))
    except Exception as e:  # no champion yet, or the site is unreachable
        print(f"(no champion loaded from {source}: {e})")
        return None


def compare_to_champion(challenger: dict, champion: dict | None, n_boot: int):
    """Paired comparison on the rows both models scored. Returns None if not comparable."""
    if champion is None:
        return None
    old = dict(zip(champion["ids"], champion["p"], strict=True))
    rows = [(y, p, old[i]) for i, y, p in zip(challenger["ids"], challenger["y"], challenger["p"], strict=True)
            if i in old]
    if len(rows) < 0.9 * len(challenger["ids"]):
        print("(champion scored a different test set; comparison not applicable)")
        return None
    y, p_new, p_old = zip(*rows, strict=True)
    result = paired_auc_diff(y, p_new, p_old, n_boot=n_boot)
    result["champion_sha"] = champion.get("git_sha", "?")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", default=CHAMPION_URL, help="URL, path, or 'none'")
    args = ap.parse_args()

    cfg = load_config()
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
    challenger = json.loads((HERE / "predictions.json").read_text(encoding="utf-8"))
    comparison = compare_to_champion(challenger, load_champion(args.champion), cfg["champion"]["n_boot"])
    checks = run_gate(metrics, cfg, comparison)

    for c in checks:
        status = "SKIP" if c.skipped else ("PASS" if c.passed else "FAIL")
        print(f"[{status}] {c.group:<26} {c.name:<34} {c.value:<32} (required {c.requirement})")

    report = {"passed": all(c.passed for c in checks), "champion_comparison": comparison,
              "checks": [c.to_dict() for c in checks]}
    (HERE / "gate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = to_markdown(checks)
    (HERE / "gate_report.md").write_text(md, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(md)

    failed = [c.name for c in checks if not c.passed]
    if failed:
        print(f"\nGATE BLOCKED: {', '.join(failed)} - this model does not ship.")
        return 1
    print("\nGATE PASSED: model quality is acceptable, deployment may proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
