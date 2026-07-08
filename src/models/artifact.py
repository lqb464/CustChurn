"""Định dạng artifact có version, metadata và hiệu chỉnh nghiệp vụ."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


@dataclass
class LendScopeArtifact:
    pipeline: Pipeline
    model_name: str
    metrics: dict[str, float]
    calibration_abs_error_q90: float
    manual_review_width_threshold: float
    max_ltv: float = 0.80
    model_version: str = "1.0.0"
    trained_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data_checksum: str = ""
    reference_statistics: dict[str, Any] = field(default_factory=dict)
    selection_results: dict[str, Any] = field(default_factory=dict)


def save_artifact(artifact: LendScopeArtifact, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination)
    return destination


def load_artifact(path: str | Path) -> LendScopeArtifact:
    artifact = joblib.load(Path(path))
    if not isinstance(artifact, LendScopeArtifact):
        raise TypeError("Artifact không đúng định dạng LendScopeArtifact")
    return artifact
