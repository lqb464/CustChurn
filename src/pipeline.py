"""Luồng huấn luyện, đánh giá holdout và suy luận end-to-end."""

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_csv, prepare_training_population, validate_schema
from src.data.preprocessor import split_train_validation_test
from src.data.schema import RAW_FEATURES, REQUEST_COLUMN, TARGET_COLUMN
from src.models.artifact import LendScopeArtifact, load_artifact, save_artifact
from src.models.evaluation import policy_baseline, regression_metrics
from src.models.trainers import select_model
from src.monitoring.drift import build_reference_statistics, detect_row_warnings


def file_checksum(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_project(
    data_path: str | Path,
    artifact_path: str | Path,
    reports_dir: str | Path,
    baseline_policy_ratio: float = 0.70,
    max_ltv: float = 0.80,
    interval_coverage: float = 0.90,
    manual_review_quantile: float = 0.90,
    shortlist_size: int = 3,
    cv_folds: int = 5,
) -> tuple[LendScopeArtifact, dict[str, float]]:
    raw = load_csv(data_path)
    population = prepare_training_population(raw)
    train, validation, holdout = split_train_validation_test(population, TARGET_COLUMN)

    selection = select_model(train, validation, top_k=shortlist_size, cv_splits=cv_folds)
    calibration_prediction = selection.pipeline.predict(validation)
    calibration_error = np.abs(validation[TARGET_COLUMN].to_numpy() - calibration_prediction)
    q90 = float(np.quantile(calibration_error, interval_coverage))
    validation_width_ratio = (2 * q90) / np.maximum(calibration_prediction, 1)
    review_threshold = float(np.quantile(validation_width_ratio, manual_review_quantile))

    train_validation = pd.concat([train, validation], ignore_index=True)
    final_pipeline = selection.pipeline
    final_pipeline.fit(train_validation, train_validation[TARGET_COLUMN])
    holdout_prediction = final_pipeline.predict(holdout)
    metrics = regression_metrics(holdout[TARGET_COLUMN], holdout_prediction)
    baseline_metrics = regression_metrics(
        holdout[TARGET_COLUMN], policy_baseline(holdout[REQUEST_COLUMN], baseline_policy_ratio)
    )
    metrics["baseline_rule70_mae"] = baseline_metrics["mae"]
    metrics["mae_improvement_vs_rule70"] = 1 - metrics["mae"] / baseline_metrics["mae"]

    artifact = LendScopeArtifact(
        pipeline=final_pipeline,
        model_name=selection.model_name,
        metrics=metrics,
        calibration_abs_error_q90=q90,
        manual_review_width_threshold=review_threshold,
        max_ltv=max_ltv,
        data_checksum=file_checksum(data_path),
        reference_statistics=build_reference_statistics(train_validation[RAW_FEATURES]),
        selection_results={
            "validation": selection.validation_results.to_dict(orient="records"),
            "cross_validation": selection.cross_validation_results.to_dict(orient="records"),
        },
    )
    holdout_scored = predict_with_artifact(artifact, holdout)
    production_prediction = holdout_scored["predicted_sanction_amount_usd"]
    metrics.update(regression_metrics(holdout[TARGET_COLUMN], production_prediction))
    actual = holdout[TARGET_COLUMN].to_numpy()
    metrics["prediction_interval_90_coverage"] = float(
        np.mean(
            (actual >= holdout_scored["prediction_lower_90_usd"].to_numpy())
            & (actual <= holdout_scored["prediction_upper_90_usd"].to_numpy())
        )
    )
    metrics["manual_review_rate"] = float(holdout_scored["manual_review"].mean())
    artifact.metrics = metrics
    save_artifact(artifact, artifact_path)

    report_path = Path(reports_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    selection.validation_results.to_csv(report_path / "validation_model_comparison.csv", index=False)
    selection.cross_validation_results.to_csv(report_path / "cross_validation_comparison.csv", index=False)
    pd.DataFrame([metrics]).to_csv(report_path / "holdout_metrics.csv", index=False)
    audit_columns = [column for column in ["Customer ID", REQUEST_COLUMN, TARGET_COLUMN] if column in holdout]
    holdout[audit_columns].join(holdout_scored).to_csv(
        report_path / "holdout_predictions.csv", index=False
    )
    return artifact, metrics


def predict_with_artifact(artifact: LendScopeArtifact, frame: pd.DataFrame) -> pd.DataFrame:
    validate_schema(frame, require_target=False)
    raw_prediction = np.asarray(artifact.pipeline.predict(frame), dtype=float)
    request = pd.to_numeric(frame[REQUEST_COLUMN], errors="coerce").to_numpy(dtype=float)
    property_price = pd.to_numeric(frame["Property Price"], errors="coerce").to_numpy(dtype=float)
    upper_business_cap = np.fmin(request, artifact.max_ltv * property_price)
    prediction = np.maximum(raw_prediction, 0)
    prediction = np.where(np.isfinite(upper_business_cap), np.minimum(prediction, upper_business_cap), prediction)
    q90 = artifact.calibration_abs_error_q90
    lower = np.maximum(prediction - q90, 0)
    upper = prediction + q90
    upper = np.where(np.isfinite(upper_business_cap), np.minimum(upper, upper_business_cap), upper)
    width_ratio = (upper - lower) / np.maximum(prediction, 1)
    warnings = detect_row_warnings(frame.reset_index(drop=True), artifact.reference_statistics)

    result = pd.DataFrame(
        {
            "predicted_sanction_amount_usd": prediction,
            "prediction_lower_90_usd": lower,
            "prediction_upper_90_usd": upper,
            "interval_width_ratio": width_ratio,
            "manual_review": width_ratio > artifact.manual_review_width_threshold,
            "input_warning": ["; ".join(items) for items in warnings],
            "model_version": artifact.model_version,
        },
        index=frame.index,
    )
    return result


def predict_file(artifact_path: str | Path, input_path: str | Path) -> pd.DataFrame:
    return predict_with_artifact(load_artifact(artifact_path), load_csv(input_path))
