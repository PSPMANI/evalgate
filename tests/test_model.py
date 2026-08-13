"""Model unit tests: the pipeline itself must behave before we spend CI time training."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from sklearn.model_selection import train_test_split  # noqa: E402

from train import SEED, build_pipeline, load_data  # noqa: E402


def _small_split():
    X, y = load_data()
    X_s, _, y_s, _ = train_test_split(X, y, train_size=1500, stratify=y, random_state=SEED)
    return X_s, y_s


def test_pipeline_trains_and_predicts_probabilities():
    X, y = _small_split()
    pipe = build_pipeline(X)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_pipeline_is_deterministic():
    X, y = _small_split()
    p1 = build_pipeline(X).fit(X, y).predict_proba(X)[:, 1]
    p2 = build_pipeline(X).fit(X, y).predict_proba(X)[:, 1]
    assert (p1 == p2).all(), "same seed must produce identical predictions"


def test_pipeline_handles_unseen_category():
    X, y = _small_split()
    pipe = build_pipeline(X)
    pipe.fit(X, y)
    X_new = X.head(5).copy()
    X_new.loc[:, "Contract"] = "BrandNewPlanType"
    proba = pipe.predict_proba(X_new)
    assert proba.shape == (5, 2), "unseen category must not crash the pipeline"
