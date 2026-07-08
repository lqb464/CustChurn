"""CLI huấn luyện artifact LendScope hoàn chỉnh."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.pipeline import train_project


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình hồi quy LendScope")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--reports-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = args.data or config["data"]["train_path"]
    artifact_path = args.artifact or config["outputs"]["artifact"]
    reports_dir = args.reports_dir or config["outputs"]["reports_dir"]
    selection = config["selection"]
    rules = config["business_rules"]
    artifact, metrics = train_project(
        data,
        artifact_path,
        reports_dir,
        baseline_policy_ratio=selection["baseline_policy_ratio"],
        max_ltv=rules["maximum_property_ltv"],
        interval_coverage=rules["prediction_interval_coverage"],
        manual_review_quantile=rules["manual_review_width_quantile"],
        shortlist_size=selection["shortlist_size"],
        cv_folds=selection["cv_folds"],
    )
    print(f"Mô hình được chọn: {artifact.model_name}")
    print(f"Artifact: {Path(artifact_path).resolve()}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
