import numpy as np

from src.data.schema import ID_COLUMNS, MODEL_FEATURES, SENSITIVE_COLUMNS
from src.features.engineering import BusinessFeatureEngineer, add_business_features


def test_business_features_are_finite_or_missing(sample_frame):
    sample_frame.loc[0, "Income (USD)"] = 0
    values = add_business_features(sample_frame)["RequestToIncome"].to_numpy()
    assert not np.isinf(values).any()


def test_production_transform_excludes_identifiers_and_gender(sample_frame):
    result = BusinessFeatureEngineer().fit_transform(sample_frame)
    assert list(result.columns) == MODEL_FEATURES
    assert not set(ID_COLUMNS + SENSITIVE_COLUMNS).intersection(result.columns)


def test_invalid_numeric_strings_become_missing(sample_frame):
    sample_frame["Property Price"] = sample_frame["Property Price"].astype(object)
    sample_frame["Co-Applicant"] = sample_frame["Co-Applicant"].astype(object)
    sample_frame.loc[0, "Property Price"] = "?"
    sample_frame.loc[0, "Co-Applicant"] = "?"
    result = BusinessFeatureEngineer().fit_transform(sample_frame)
    assert np.isnan(result.loc[0, "Property Price"])
    assert np.isnan(result.loc[0, "RequestToProperty"])
