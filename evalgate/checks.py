"""The release policy as code: each check returns a structured pass/fail result.

Keeping the checks as pure functions over (metrics, config) means the gate that
blocks deployments is itself unit-tested on hand-built inputs.
"""
from __future__ import annotations

import pathlib
import tomllib
from dataclasses import asdict, dataclass

ROOT = pathlib.Path(__file__).resolve().parent.parent


@dataclass
class Check:
    group: str
    name: str
    passed: bool
    value: str
    requirement: str
    skipped: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: pathlib.Path = ROOT / "gate.toml") -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def threshold_checks(m: dict, cfg: dict) -> list[Check]:
    t = cfg["thresholds"]
    out = [Check("quality", key, m[key] >= t[key], f"{m[key]:.4f}", f">= {t[key]}")
           for key in ("roc_auc_test", "accuracy_test", "roc_auc_cv_mean")]
    gap = abs(m["roc_auc_test"] - m["roc_auc_cv_mean"])
    out.append(Check("quality", "cv_test_gap", gap <= t["max_cv_test_gap"], f"{gap:.4f}",
                     f"<= {t['max_cv_test_gap']}"))
    return out


def calibration_checks(m: dict, cfg: dict) -> list[Check]:
    c = cfg["calibration"]
    return [Check("calibration", "ece", m["ece"] <= c["max_ece"], f"{m['ece']:.4f}", f"<= {c['max_ece']}"),
            Check("calibration", "brier", m["brier"] <= c["max_brier"], f"{m['brier']:.4f}",
                  f"<= {c['max_brier']}")]


def slice_checks(m: dict, cfg: dict) -> list[Check]:
    s = cfg["slices"]
    out = []
    for col, groups in m["slices"].items():
        for value, g in groups.items():
            if g["n"] < s["min_slice_n"] or g["auc"] is None:
                out.append(Check("slices", f"{col}={value}", True, f"n={g['n']}",
                                 f"n >= {s['min_slice_n']} to gate", skipped=True))
                continue
            out.append(Check("slices", f"{col}={value}", g["auc"] >= s["min_slice_auc"],
                             f"{g['auc']:.4f} (n={g['n']})", f">= {s['min_slice_auc']}"))
    for col in s["fairness_columns"]:
        aucs = [g["auc"] for g in m["slices"].get(col, {}).values() if g["auc"] is not None]
        if len(aucs) >= 2:
            gap = max(aucs) - min(aucs)
            out.append(Check("fairness", f"{col} AUC gap", gap <= s["max_fairness_gap"], f"{gap:.4f}",
                             f"<= {s['max_fairness_gap']}"))
    return out


def champion_check(comparison: dict | None, cfg: dict) -> list[Check]:
    margin = cfg["champion"]["non_inferiority_margin"]
    if comparison is None:
        return [Check("champion", "non-inferior to deployed model", True, "no deployed champion",
                      f"lower 95% bound of AUC diff >= -{margin}", skipped=True)]
    lo = comparison["ci"][0]
    return [Check("champion", "non-inferior to deployed model", lo >= -margin,
                  f"{comparison['diff']:+.4f} [{comparison['ci'][0]:+.4f}, {comparison['ci'][1]:+.4f}]",
                  f"lower 95% bound >= -{margin}")]


def drift_checks(psi_by_feature: dict, cfg: dict, group: str = "drift") -> list[Check]:
    d = cfg["drift"]
    return [Check(group, f"PSI {feat}", v <= d["psi_block"], f"{v:.4f}", f"<= {d['psi_block']}")
            for feat, v in sorted(psi_by_feature.items(), key=lambda kv: -kv[1])]


def run_gate(metrics: dict, cfg: dict, comparison: dict | None = None) -> list[Check]:
    return (threshold_checks(metrics, cfg) + calibration_checks(metrics, cfg)
            + slice_checks(metrics, cfg) + champion_check(comparison, cfg)
            + drift_checks(metrics.get("drift_test_vs_train", {}), cfg, "drift (held-out vs train)"))


def to_markdown(checks: list[Check], title: str = "EvalGate") -> str:
    failed = [c for c in checks if not c.passed]
    verdict = "BLOCKED" if failed else "PASSED"
    lines = [f"## {title}: {verdict}", "",
             f"{len(checks)} checks, {len(failed)} failed, {sum(c.skipped for c in checks)} skipped.", "",
             "| Group | Check | Value | Requirement | Result |", "|---|---|---|---|---|"]
    for c in checks:
        result = "skipped" if c.skipped else ("PASS" if c.passed else "**FAIL**")
        lines.append(f"| {c.group} | {c.name} | {c.value} | {c.requirement} | {result} |")
    return "\n".join(lines) + "\n"
