"""Project configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


@dataclass(frozen=True)
class ModelConfig:
    alias: str
    hf_name: str
    local_path: Path
    family: str
    hidden_size: int
    vocab_size: int
    recommended_dtype: str = "bfloat16"
    tie_word_embeddings: bool | None = None
    raw: dict[str, Any] | None = None


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def load_config(name: str) -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / name)


def model_configs(path: str | Path | None = None) -> dict[str, ModelConfig]:
    data = load_yaml(path or CONFIG_DIR / "models.yaml")
    entries = data.get("models", {})
    configs: dict[str, ModelConfig] = {}
    for alias, raw in entries.items():
        configs[alias] = ModelConfig(
            alias=alias,
            hf_name=str(raw["hf_name"]),
            local_path=Path(raw["local_path"]).expanduser(),
            family=str(raw.get("family", "unknown")),
            hidden_size=int(raw["hidden_size"]),
            vocab_size=int(raw["vocab_size"]),
            recommended_dtype=str(raw.get("recommended_dtype", "bfloat16")),
            tie_word_embeddings=raw.get("tie_word_embeddings"),
            raw=dict(raw),
        )
    return configs


def resolve_model(model: str, path: str | Path | None = None) -> ModelConfig:
    configs = model_configs(path)
    if model in configs:
        return configs[model]
    for alias, cfg in configs.items():
        if model in {cfg.hf_name, str(cfg.local_path)}:
            return cfg
    candidate = Path(model).expanduser()
    if candidate.exists():
        cfg_path = candidate / "config.json"
        hidden_size = 0
        vocab_size = 0
        family = "unknown"
        if cfg_path.exists():
            import json

            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            hidden_size = int(raw.get("hidden_size", 0))
            vocab_size = int(raw.get("vocab_size", 0))
            family = str(raw.get("model_type", "unknown"))
        return ModelConfig(
            alias=candidate.name,
            hf_name=str(candidate),
            local_path=candidate,
            family=family,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )
    known = ", ".join(sorted(configs))
    raise KeyError(f"Unknown model {model!r}. Known aliases: {known}")

