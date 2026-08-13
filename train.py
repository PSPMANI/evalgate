"""Train the churn model and write the metrics the pipeline gates on.

    python train.py

Produces:
  model.joblib   - the trained sklearn Pipeline (the artifact CI promotes)
  metrics.json   - the numbers gate.py enforces and the dashboard displays

Deterministic (fixed seeds) so CI results are reproducible.
"""
import json
import pathlib
import subprocess
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data" / "telco_churn.csv"
SEED = 42


def load_data():
    df = pd.read_csv(DATA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn", "customerID"])
    return X, y


def build_pipeline(X):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", GradientBoostingClassifier(random_state=SEED)),
    ])


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True
        ).strip()
    except Exception:
        return "local"


def main():
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    pipe = build_pipeline(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")

    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc_test": round(float(roc_auc_score(y_te, proba)), 4),
        "accuracy_test": round(float(accuracy_score(y_te, pred)), 4),
        "roc_auc_cv_mean": round(float(cv_scores.mean()), 4),
        "roc_auc_cv_std": round(float(cv_scores.std()), 4),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "git_sha": git_sha(),
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }

    joblib.dump(pipe, HERE / "model.joblib")
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("wrote model.joblib and metrics.json")


if __name__ == "__main__":
    main()
