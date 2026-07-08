import pytest

from src.models.evaluation import policy_baseline, regression_metrics


def test_regression_metrics_have_business_semantics():
    metrics = regression_metrics([100, 200], [110, 180])
    assert metrics["mae"] == pytest.approx(15)
    assert metrics["bias"] == pytest.approx(-5)
    assert metrics["over_prediction_rate"] == pytest.approx(0.5)


def test_rule70_baseline():
    assert policy_baseline([100, 200]).tolist() == pytest.approx([70, 140])
