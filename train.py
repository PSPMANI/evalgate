"""Train the churn model and write everything the release gate needs.

    python train.py              # the real run
    python train.py --sabotage   # negative control: shuffled labels, must be BLOCKED
    python train.py --n-estimators 20 --max-depth 2   # a subtly weaker challenger

Produces:
  model.joblib       - the trained sklearn Pipeline (the artifact CI promotes)
  metrics.json       - quality, calibration, slice and drift numbers gate.py enforces
  predictions.json   - held-out ids, labels and scores (the next run's champion baseline)
  data_profile.json  - reference feature distributions for drift monitoring

Deterministic (fixed seeds) so CI results are reproducible.
"""
import argparse
import json
import pathlib
import subprocess
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evalgate.checks import load_config
from evalgate.metrics import auc_ci, drift, ece, profile, reliability_curve, slice_metrics

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data" / "telco_churn.csv"
SEED = 42


def load_data(with_ids: bool = False):
    df = pd.read_csv(DATA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn", "customerID"])
    if with_ids:
        return X, y, df["customerID"]
    return X, y


def build_pipeline(X, **clf_params):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", GradientBoostingClassifier(random_state=SEED, **clf_params)),
    ])


def tenure_band(tenure: pd.Series) -> pd.Series:
    return pd.cut(tenure, [-1, 12, 24, 48, 10_000], labels=["0-12m", "13-24m", "25-48m", "49m+"])


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True
        ).strip()
    except Exception:
        return "local"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sabotage", action="store_true",
                    help="shuffle the training labels; the gate must block the result")
    ap.add_argument("--n-estimators", type=int, help="override the booster size (regression demo)")
    ap.add_argument("--max-depth", type=int, help="override the tree depth (regression demo)")
    args = ap.parse_args()
    clf_params = {k: v for k, v in (("n_estimators", args.n_estimators), ("max_depth", args.max_depth))
                  if v is not None}
    cfg = load_config()

    X, y, ids = load_data(with_ids=True)
    X_tr, X_te, y_tr, y_te, _, id_te = train_test_split(
        X, y, ids, test_size=0.2, stratify=y, random_state=SEED)
    if args.sabotage:
        y_tr = pd.Series(np.random.default_rng(SEED).permutation(y_tr.values), index=y_tr.index)

    pipe = build_pipeline(X, **clf_params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")

    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    frame = X_te.copy()
    frame["tenure_band"] = tenure_band(frame["tenure"])
    frame["y"], frame["p"] = y_te.values, proba
    ref_profile = profile(X_tr)

    metrics = {
        "roc_auc_test": round(float(roc_auc_score(y_te, proba)), 4),
        "roc_auc_test_ci": auc_ci(y_te.values, proba),
        "accuracy_test": round(float(accuracy_score(y_te, pred)), 4),
        "roc_auc_cv_mean": round(float(cv_scores.mean()), 4),
        "roc_auc_cv_std": round(float(cv_scores.std()), 4),
        "brier": round(float(brier_score_loss(y_te, proba)), 4),
        "ece": round(ece(y_te.values, proba), 4),
        "reliability": reliability_curve(y_te.values, proba),
        "slices": slice_metrics(frame, cfg["slices"]["columns"]),
        "drift_test_vs_train": drift(ref_profile, X_te),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "sabotaged": bool(args.sabotage),
        "git_sha": git_sha(),
        "trained_at_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
    }

    joblib.dump(pipe, HERE / "model.joblib")
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (HERE / "predictions.json").write_text(json.dumps({
        "git_sha": metrics["git_sha"], "trained_at_utc": metrics["trained_at_utc"],
        "ids": id_te.tolist(), "y": y_te.astype(int).tolist(),
        "p": [round(float(x), 6) for x in proba]}), encoding="utf-8")
    (HERE / "data_profile.json").write_text(json.dumps(ref_profile, indent=2), encoding="utf-8")
    summary = {k: v for k, v in metrics.items() if k not in ("slices", "reliability", "drift_test_vs_train")}
    print(json.dumps(summary, indent=2))
    print("wrote model.joblib, metrics.json, predictions.json, data_profile.json")


if __name__ == "__main__":
    main()
