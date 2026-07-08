"""Đọc, kiểm tra và xác định population hợp lệ để huấn luyện."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import (
    ID_COLUMNS,
    RAW_FEATURES,
    SENTINEL_VALUE,
    SENSITIVE_COLUMNS,
    TARGET_COLUMN,
)


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {path}")
    return pd.read_csv(path)


def validate_schema(frame: pd.DataFrame, require_target: bool = True) -> None:
    required = set(RAW_FEATURES)
    if require_target:
        required.add(TARGET_COLUMN)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Thiếu các cột bắt buộc: {missing}")


def normalize_sentinels(frame: pd.DataFrame) -> pd.DataFrame:
    """Đổi mã thiếu -999 thành NaN; không sửa DataFrame đầu vào."""
    normalized = frame.copy()
    return normalized.replace(SENTINEL_VALUE, np.nan)


def valid_regression_mask(frame: pd.DataFrame) -> pd.Series:
    """Bài toán có điều kiện: chỉ khoản vay lịch sử đã được duyệt dương."""
    if TARGET_COLUMN not in frame:
        raise ValueError(f"Không có cột mục tiêu {TARGET_COLUMN!r}")
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    return target.gt(0) & target.notna()


def prepare_training_population(frame: pd.DataFrame) -> pd.DataFrame:
    validate_schema(frame, require_target=True)
    normalized = normalize_sentinels(frame)
    population = normalized.loc[valid_regression_mask(normalized)].copy()
    population[TARGET_COLUMN] = pd.to_numeric(population[TARGET_COLUMN])
    return population.reset_index(drop=True)


def data_quality_summary(frame: pd.DataFrame) -> dict[str, object]:
    target = (
        pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        if TARGET_COLUMN in frame
        else None
    )
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "valid_positive_target_rows": int((target > 0).sum()) if target is not None else None,
        "zero_target_rows": int((target == 0).sum()) if target is not None else None,
        "invalid_target_rows": int((target < 0).sum()) if target is not None else None,
        "missing_target_rows": int(target.isna().sum()) if target is not None else None,
        "identifier_columns_present": [c for c in ID_COLUMNS if c in frame],
        "sensitive_columns_present": [c for c in SENSITIVE_COLUMNS if c in frame],
    }
