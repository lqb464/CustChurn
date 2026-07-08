from sklearn.linear_model import LinearRegression

from src.data.schema import RAW_FEATURES, TARGET_COLUMN
from src.models.artifact import LendScopeArtifact
from src.models.trainers import build_model_pipeline
from src.monitoring.drift import build_reference_statistics
from src.pipeline import predict_with_artifact


def test_inference_applies_caps_and_returns_review_fields(sample_frame):
    pipeline = build_model_pipeline(LinearRegression(), scale_numeric=True)
    pipeline.fit(sample_frame, sample_frame[TARGET_COLUMN])
    artifact = LendScopeArtifact(
        pipeline, "linear", {}, 2000, 0.10,
        reference_statistics=build_reference_statistics(sample_frame[RAW_FEATURES]),
    )
    result = predict_with_artifact(artifact, sample_frame.head(5))
    assert (result["predicted_sanction_amount_usd"] >= 0).all()
    assert (result["predicted_sanction_amount_usd"] <= sample_frame.head(5)["Loan Amount Request (USD)"]).all()
    assert {"manual_review", "input_warning", "model_version"}.issubset(result.columns)


def test_inference_warns_for_invalid_numeric_string(sample_frame):
    pipeline = build_model_pipeline(LinearRegression(), scale_numeric=True)
    pipeline.fit(sample_frame, sample_frame[TARGET_COLUMN])
    artifact = LendScopeArtifact(
        pipeline, "linear", {}, 2000, 0.10,
        reference_statistics=build_reference_statistics(sample_frame[RAW_FEATURES]),
    )
    invalid = sample_frame.head(1).copy()
    invalid["Property Price"] = "?"
    result = predict_with_artifact(artifact, invalid)
    assert "không phải giá trị số hợp lệ" in result.iloc[0]["input_warning"]
