import numpy as np
import pytest

from src.data.loader import normalize_sentinels, prepare_training_population, validate_schema
from src.data.schema import TARGET_COLUMN


def test_prepare_population_keeps_only_positive_target(sample_frame):
    sample_frame.loc[0, TARGET_COLUMN] = 0
    sample_frame.loc[1, TARGET_COLUMN] = -999
    sample_frame.loc[2, TARGET_COLUMN] = np.nan
    result = prepare_training_population(sample_frame)
    assert len(result) == len(sample_frame) - 3
    assert result[TARGET_COLUMN].gt(0).all()


def test_sentinel_normalization_does_not_mutate_input(sample_frame):
    sample_frame.loc[0, "Dependents"] = -999
    result = normalize_sentinels(sample_frame)
    assert sample_frame.loc[0, "Dependents"] == -999
    assert np.isnan(result.loc[0, "Dependents"])


def test_schema_reports_missing_columns(sample_frame):
    with pytest.raises(ValueError, match="Thiếu"):
        validate_schema(sample_frame.drop(columns="Credit Score"))


def test_quality_summary_supports_inference_without_target(sample_frame):
    from src.data.loader import data_quality_summary

    summary = data_quality_summary(sample_frame.drop(columns=TARGET_COLUMN))
    assert summary["missing_target_rows"] is None
