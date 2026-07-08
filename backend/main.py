"""API dự báo khoản vay với schema validation từ pipeline dùng chung."""

import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.artifact import LendScopeArtifact, load_artifact
from src.pipeline import predict_with_artifact

DEFAULT_ARTIFACT_PATH = Path("outputs/models/lendscope.joblib")


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1, max_length=1000)


app = FastAPI(
    title="LendScope API",
    version="1.0.0",
    description="Ước lượng số tiền được phê duyệt cho hồ sơ đã qua bước tiền thẩm định.",
)
_artifact: LendScopeArtifact | None = None


def artifact_path() -> Path:
    return Path(os.getenv("LENDSCOPE_MODEL_PATH", str(DEFAULT_ARTIFACT_PATH)))


def get_artifact() -> LendScopeArtifact:
    global _artifact
    if _artifact is None:
        path = artifact_path()
        if not path.exists():
            raise HTTPException(status_code=503, detail=f"Chưa có artifact tại {path}")
        _artifact = load_artifact(path)
    return _artifact


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "artifact_available": artifact_path().exists()}


@app.get("/model")
def model_info() -> dict[str, object]:
    artifact = get_artifact()
    return {
        "model_name": artifact.model_name,
        "model_version": artifact.model_version,
        "trained_at_utc": artifact.trained_at_utc,
        "metrics": artifact.metrics,
        "calibration_abs_error_q90": artifact.calibration_abs_error_q90,
    }


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, object]:
    try:
        frame = pd.DataFrame(payload.records)
        result = predict_with_artifact(get_artifact(), frame)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"count": len(result), "predictions": result.to_dict(orient="records")}
