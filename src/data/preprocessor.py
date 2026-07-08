"""Tách dữ liệu và xây dựng preprocessing tránh leakage."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.data.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE


def quantile_labels(target: pd.Series, bins: int = 10) -> pd.Series:
    """Nhãn phân tầng bền vững khi target có nhiều giá trị trùng."""
    ranks = target.rank(method="first")
    return pd.qcut(ranks, q=min(bins, len(target)), labels=False, duplicates="drop")


def split_train_validation_test(
    frame: pd.DataFrame,
    target_column: str,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tách 70/15/15 và giữ phân phối target gần tương đương."""
    train, temporary = train_test_split(
        frame,
        test_size=0.30,
        random_state=random_state,
        stratify=quantile_labels(frame[target_column]),
    )
    validation, test = train_test_split(
        temporary,
        test_size=0.50,
        random_state=random_state,
        stratify=quantile_labels(temporary[target_column]),
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", RobustScaler()))

    numeric = Pipeline(numeric_steps)
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
