"""Data-contract tests: CI fails if the training data breaks its schema."""
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from train import load_data  # noqa: E402

REQUIRED_COLUMNS = {
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "Contract", "MonthlyCharges", "TotalCharges",
}


def test_required_columns_present():
    X, _ = load_data()
    missing = REQUIRED_COLUMNS - set(X.columns)
    assert not missing, f"schema broken, missing columns: {missing}"


def test_target_is_binary_and_nonempty():
    _, y = load_data()
    assert len(y) > 5000, "dataset shrank unexpectedly"
    assert set(y.unique()) == {0, 1}


def test_no_identifier_leakage():
    X, _ = load_data()
    assert "customerID" not in X.columns, "identifier column leaked into features"
    assert "Churn" not in X.columns, "target leaked into features"


def test_no_missing_numeric_values():
    X, _ = load_data()
    num = X.select_dtypes(include="number")
    assert not num.isna().any().any(), "numeric features contain NaN after cleaning"
