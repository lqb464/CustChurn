"""CLI dự báo theo lô từ artifact đã huấn luyện."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.loader import load_csv
from src.models.artifact import load_artifact
from src.pipeline import predict_with_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dự báo khoản vay được phê duyệt")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--output", default="outputs/reports/test_predictions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    input_path = args.input or config["data"]["inference_path"]
    artifact_path = args.artifact or config["outputs"]["artifact"]
    frame = load_csv(input_path)
    result = predict_with_artifact(load_artifact(artifact_path), frame)
    identifiers = [c for c in ["Customer ID", "Name", "Property ID"] if c in frame]
    output = frame[identifiers].join(result)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    print(f"Đã ghi {len(output):,} dự báo vào {destination.resolve()}")
    print(f"Số hồ sơ cần rà soát thủ công: {int(result['manual_review'].sum()):,}")


if __name__ == "__main__":
    main()
