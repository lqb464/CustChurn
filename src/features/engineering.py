"""Feature engineering theo các quan hệ tài chính có thể giải thích."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.data.schema import ENGINEERED_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, RAW_FEATURES, REQUEST_COLUMN


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def add_business_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy().replace(-999, np.nan)
    raw_numeric = [column for column in NUMERIC_FEATURES if column not in ENGINEERED_FEATURES]
    for column in raw_numeric:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["RequestToIncome"] = _safe_divide(output[REQUEST_COLUMN], output["Income (USD)"])
    output["RequestToProperty"] = _safe_divide(output[REQUEST_COLUMN], output["Property Price"])
    output["ExpenseToIncome"] = _safe_divide(
        output["Current Loan Expenses (USD)"], output["Income (USD)"]
    )
    output["IncomePerDependent"] = _safe_divide(
        output["Income (USD)"], output["Dependents"].fillna(0) + 1
    )
    output["PropertyEquityProxy"] = output["Property Price"] - output[REQUEST_COLUMN]
    output["CreditIncomeInteraction"] = output["Credit Score"] * output["Income (USD)"]
    output["CreditRequestInteraction"] = output["Credit Score"] * output[REQUEST_COLUMN]
    output["AgeIncomeInteraction"] = output["Age"] * output["Income (USD)"]
    return output.replace([np.inf, -np.inf], np.nan)


class BusinessFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformer sklearn nhận dữ liệu raw và chỉ trả feature sản xuất."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "BusinessFeatureEngineer":
        missing = sorted(set(RAW_FEATURES).difference(X.columns))
        if missing:
            raise ValueError(f"Thiếu feature đầu vào: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_business_features(X.loc[:, RAW_FEATURES]).loc[:, MODEL_FEATURES]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(MODEL_FEATURES, dtype=object)
