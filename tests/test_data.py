import pandas as pd
import pytest

from fraud_detection.data import FEATURE_COLUMNS, TARGET_COLUMN, make_synthetic, validate


def test_synthetic_schema():
    df = make_synthetic(n_rows=1_000, seed=1)
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert len(df) == 1_000


def test_synthetic_is_imbalanced_but_has_fraud():
    df = make_synthetic(n_rows=5_000, fraud_rate=0.01, seed=2)
    rate = df[TARGET_COLUMN].mean()
    assert 0 < rate < 0.05


def test_synthetic_reproducible():
    a = make_synthetic(n_rows=500, seed=3)
    b = make_synthetic(n_rows=500, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_validate_passes_on_synthetic():
    validate(make_synthetic(n_rows=500))


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda d: d.drop(columns=["V1"]), "Missing columns"),
        (lambda d: d.assign(Amount=-1.0), "Negative"),
        (lambda d: d.assign(Class=0), "No positive"),
        (lambda d: d.assign(V2=float("nan")), "NaNs"),
    ],
)
def test_validate_catches_bad_data(mutation, match):
    df = mutation(make_synthetic(n_rows=500))
    with pytest.raises(ValueError, match=match):
        validate(df)
