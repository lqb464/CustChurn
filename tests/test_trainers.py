from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.data.schema import TARGET_COLUMN
from src.models.trainers import build_model_pipeline


def test_pipeline_accepts_raw_frame(sample_frame):
    pipeline = build_model_pipeline(LinearRegression(), scale_numeric=True)
    pipeline.fit(sample_frame, sample_frame[TARGET_COLUMN])
    assert pipeline.predict(sample_frame.head(3)).shape == (3,)


def test_tree_pipeline_handles_unknown_category(sample_frame):
    pipeline = build_model_pipeline(RandomForestRegressor(n_estimators=5, random_state=42))
    pipeline.fit(sample_frame, sample_frame[TARGET_COLUMN])
    unknown = sample_frame.head(1).copy()
    unknown["Profession"] = "Nhom moi"
    assert pipeline.predict(unknown).shape == (1,)
