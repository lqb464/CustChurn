from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

import backend.main as api
from src.data.schema import RAW_FEATURES, TARGET_COLUMN
from src.models.artifact import LendScopeArtifact
from src.models.trainers import build_model_pipeline
from src.monitoring.drift import build_reference_statistics


def test_health_and_predict_endpoints(sample_frame, monkeypatch):
    pipeline = build_model_pipeline(LinearRegression(), scale_numeric=True)
    pipeline.fit(sample_frame, sample_frame[TARGET_COLUMN])
    artifact = LendScopeArtifact(
        pipeline, "linear", {}, 2000, 0.1,
        reference_statistics=build_reference_statistics(sample_frame[RAW_FEATURES]),
    )
    monkeypatch.setattr(api, "_artifact", artifact)
    client = TestClient(api.app)
    assert client.get("/health").status_code == 200
    records = sample_frame.head(2).where(sample_frame.head(2).notna(), None).to_dict(orient="records")
    response = client.post("/predict", json={"records": records})
    assert response.status_code == 200
    assert response.json()["count"] == 2
