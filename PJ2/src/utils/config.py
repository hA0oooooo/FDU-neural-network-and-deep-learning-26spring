from __future__ import annotations

import copy
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment_configs(
    config_dir: str | Path,
    ordered_ids: list[str] | None = None,
) -> list[dict]:
    paths = sorted(Path(config_dir).glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"No YAML config files found in {config_dir}")
    configs = [load_config(path) for path in paths]
    if ordered_ids is None:
        return configs

    by_id = {cfg["id"]: cfg for cfg in configs}
    missing = [exp_id for exp_id in ordered_ids if exp_id not in by_id]
    if missing:
        raise ValueError(f"Missing config files for experiments: {missing}")
    return [by_id[exp_id] for exp_id in ordered_ids]


def resolve_config_path(config: str | Path, config_dir: str | Path) -> Path:
    path = Path(config)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return Path(config_dir) / path


def load_selected_configs(
    config_dir: str | Path,
    ordered_ids: list[str],
    config: str | Path | list[str | Path] | None = None,
) -> list[dict]:
    if config is None:
        return load_experiment_configs(config_dir, ordered_ids)

    if isinstance(config, (str, Path)):
        configs = [config]
    else:
        configs = config

    return [load_config(resolve_config_path(path, config_dir)) for path in configs]


def build_experiment_config(exp_cfg: dict) -> dict:
    return {
        "model": copy.deepcopy(exp_cfg["model"]),
        "train": copy.deepcopy(exp_cfg["train"]),
    }
