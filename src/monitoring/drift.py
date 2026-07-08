"""Giám sát phạm vi dữ liệu nhẹ, không phụ thuộc nền tảng triển khai."""

import numpy as np
import pandas as pd

from src.data.schema import CATEGORICAL_FEATURES, RAW_FEATURES


def build_reference_statistics(frame: pd.DataFrame) -> dict[str, object]:
    statistics: dict[str, object] = {"rows": int(len(frame)), "numeric": {}, "categorical": {}}
    for column in RAW_FEATURES:
        if column not in frame:
            continue
        if column in CATEGORICAL_FEATURES:
            values = frame[column].dropna().astype(str)
            statistics["categorical"][column] = {
                "known_values": sorted(values.unique().tolist()),
                "missing_rate": float(frame[column].isna().mean()),
            }
        else:
            values = pd.to_numeric(frame[column], errors="coerce")
            statistics["numeric"][column] = {
                "p01": float(values.quantile(0.01)),
                "p50": float(values.quantile(0.50)),
                "p99": float(values.quantile(0.99)),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "missing_rate": float(values.isna().mean()),
            }
    return statistics


def detect_row_warnings(frame: pd.DataFrame, reference: dict[str, object]) -> list[list[str]]:
    warnings: list[list[str]] = [[] for _ in range(len(frame))]
    numeric = reference.get("numeric", {})
    categorical = reference.get("categorical", {})
    for column, stats in numeric.items():
        if column not in frame:
            continue
        raw_values = frame[column]
        values = pd.to_numeric(raw_values, errors="coerce")
        invalid = raw_values.notna() & values.isna()
        for position in np.flatnonzero(invalid.to_numpy()):
            warnings[position].append(f"{column} không phải giá trị số hợp lệ và đã được xem là thiếu")
        outside = values.lt(stats["p01"]) | values.gt(stats["p99"])
        for position in np.flatnonzero(outside.to_numpy()):
            warnings[position].append(f"{column} nằm ngoài khoảng p01-p99 của dữ liệu huấn luyện")
    for column, stats in categorical.items():
        if column not in frame:
            continue
        known = set(stats["known_values"])
        unseen = frame[column].notna() & ~frame[column].astype(str).isin(known)
        for position in np.flatnonzero(unseen.to_numpy()):
            warnings[position].append(f"{column} có nhóm chưa từng xuất hiện khi huấn luyện")
    return warnings


def population_drift_report(frame: pd.DataFrame, reference: dict[str, object]) -> pd.DataFrame:
    """Báo cáo thay đổi trung bình và tỷ lệ thiếu để phục vụ cảnh báo vận hành."""
    rows: list[dict[str, object]] = []
    for column, stats in reference.get("numeric", {}).items():
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        scale = stats["std"] or 1.0
        rows.append(
            {
                "feature": column,
                "standardized_mean_shift": float(abs(values.mean() - stats["mean"]) / scale),
                "missing_rate_shift": float(values.isna().mean() - stats["missing_rate"]),
            }
        )
    return pd.DataFrame(rows).sort_values("standardized_mean_shift", ascending=False).reset_index(drop=True)
