import numpy as np
import pytest

from fraud_detection.data import make_synthetic
from fraud_detection.drift import DRIFT_THRESHOLD, drift_report, psi


def test_psi_identical_distributions_near_zero():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=5_000), rng.normal(size=5_000)
    assert psi(a, b) < 0.05


def test_psi_shifted_distribution_flags_drift():
    rng = np.random.default_rng(1)
    ref, cur = rng.normal(0, 1, 5_000), rng.normal(2, 1, 5_000)
    assert psi(ref, cur) > DRIFT_THRESHOLD


def test_psi_constant_feature_handled():
    assert psi(np.ones(100), np.ones(100)) == 0.0
    assert psi(np.ones(100), np.full(100, 5.0)) == float("inf")

def test_psi_empty_raises():
    with pytest.raises(ValueError):
        psi(np.array([]), np.array([1.0]))


def test_drift_report_no_drift_on_same_seedless_sample():
    ref = make_synthetic(4_000, seed=10)
    cur = make_synthetic(4_000, seed=20)  # same distribution, different draw
    report = drift_report(ref, cur)
    assert (report["status"] == "drift").sum() == 0


def test_drift_report_detects_injected_shift():
    ref = make_synthetic(4_000, seed=10)
    cur = make_synthetic(4_000, seed=20)
    cur["V1"] = cur["V1"] + 3.0  # inject drift into one feature
    report = drift_report(ref, cur)
    assert report.iloc[0]["feature"] == "V1"
    assert report.iloc[0]["status"] == "drift"
