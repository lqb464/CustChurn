"""Kiểm tra schema và chất lượng dữ liệu trước khi huấn luyện."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.loader import data_quality_summary, load_csv, validate_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra dữ liệu LendScope")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--inference", action="store_true", help="Không yêu cầu cột target")
    args = parser.parse_args()
    config = load_config(args.config)
    default_path = config["data"]["inference_path" if args.inference else "train_path"]
    frame = load_csv(args.data or default_path)
    validate_schema(frame, require_target=not args.inference)
    print(json.dumps(data_quality_summary(frame), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
