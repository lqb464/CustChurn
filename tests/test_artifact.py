from sklearn.linear_model import LinearRegression

from src.data.schema import TARGET_COLUMN
from src.models.artifact import LendScopeArtifact, load_artifact, save_artifact
from src.models.trainers import build_model_pipeline


def test_artifact_roundtrip(sample_frame, tmp_path):
    pipeline = build_model_pipeline(LinearRegression(), scale_numeric=True)
    pipeline.fit(sample_frame, sample_frame[TARGET_COLUMN])
    artifact = LendScopeArtifact(pipeline, "linear", {}, 1000, 0.5)
    loaded = load_artifact(save_artifact(artifact, tmp_path / "model.joblib"))
    assert loaded.model_name == "linear"
    assert loaded.pipeline.predict(sample_frame.head(1)).shape == (1,)
