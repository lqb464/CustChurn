"""Đọc cấu hình YAML với thông báo lỗi rõ ràng."""

from pathlib import Path

import yaml


def load_config(path: str | Path = "configs/config.yaml") -> dict[str, object]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy cấu hình: {config_path}")
    with config_path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError("Cấu hình YAML phải là một mapping")
    return config
