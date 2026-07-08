"""Ứng viên mô hình, lựa chọn bằng validation kết hợp cross-validation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.data.preprocessor import build_preprocessor
from src.data.schema import RANDOM_STATE, TARGET_COLUMN
from src.features.engineering import BusinessFeatureEngineer
from src.models.evaluation import regression_metrics


def candidate_estimators(random_state: int = RANDOM_STATE) -> dict[str, object]:
    candidates: dict[str, object] = {
        "LinearRegression": LinearRegression(),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.10, max_iter=10_000),
        "RandomForest": RandomForestRegressor(
            n_estimators=350,
            min_samples_leaf=2,
            max_features=0.85,
            n_jobs=-1,
            random_state=random_state,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=350,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=random_state,
        ),
    }
    try:
        from xgboost import XGBRegressor

        candidates["XGBoost"] = XGBRegressor(
            n_estimators=550,
            learning_rate=0.04,
            max_depth=6,
            min_child_weight=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=random_state,
        )
    except ImportError:
        pass
    return candidates


def build_model_pipeline(estimator: object, scale_numeric: bool = False) -> Pipeline:
    return Pipeline(
        [
            ("features", BusinessFeatureEngineer()),
            ("preprocessor", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )


@dataclass
class SelectionResult:
    model_name: str
    pipeline: Pipeline
    validation_results: pd.DataFrame
    cross_validation_results: pd.DataFrame


def select_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    top_k: int = 3,
    cv_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> SelectionResult:
    y_train = train[TARGET_COLUMN]
    y_validation = validation[TARGET_COLUMN]
    fitted: dict[str, Pipeline] = {}
    validation_rows: list[dict[str, object]] = []

    for name, estimator in candidate_estimators(random_state).items():
        pipeline = build_model_pipeline(estimator, scale_numeric=name in {"LinearRegression", "ElasticNet"})
        pipeline.fit(train, y_train)
        fitted[name] = pipeline
        validation_rows.append({"model": name, **regression_metrics(y_validation, pipeline.predict(validation))})

    validation_results = pd.DataFrame(validation_rows).sort_values("mae").reset_index(drop=True)
    top_names = validation_results.head(top_k)["model"].tolist()
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    cv_rows: list[dict[str, object]] = []
    for name in top_names:
        scores = -cross_val_score(
            clone(fitted[name]),
            train,
            y_train,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=1,
        )
        cv_rows.append(
            {
                "model": name,
                "cv_mae_mean": float(scores.mean()),
                "cv_mae_std": float(scores.std()),
            }
        )
    cv_results = pd.DataFrame(cv_rows).sort_values("cv_mae_mean").reset_index(drop=True)
    winner = str(cv_results.iloc[0]["model"])
    return SelectionResult(winner, fitted[winner], validation_results, cv_results)
