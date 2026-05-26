from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_DIR = ROOT / "configs" / "bn"
EXPERIMENT_ORDER = [
    "vgga_lr_1e_4",
    "vgga_lr_5e_4",
    "vgga_lr_1e_3",
    "vgga_lr_2e_3",
    "vgga_lr_5e_3",
    "vgga_bn_lr_1e_4",
    "vgga_bn_lr_5e_4",
    "vgga_bn_lr_1e_3",
    "vgga_bn_lr_2e_3",
    "vgga_bn_lr_5e_3",
]

from src.models.vgg import VGG_A, VGG_A_BatchNorm
from src.utils.config import load_selected_configs
from src.utils.data import CIFAR10_CLASSES, build_cifar10_train_test_loaders
from src.utils.plot import plot_bn_analysis_figures
from src.utils.seed import get_device, set_seed
from src.utils.train import fit_step_experiment


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    return parser.parse_args()


def remove_root_outputs(output_dir: Path) -> None:
    for name in [
        "loss_landscape.png",
        "loss_landscape_batch.png",
        "loss_landscape_epoch.png",
        "gradient_predictiveness.png",
        "effective_beta_smoothness.png",
        "sam_sharpness.png",
        "analysis_metrics.csv",
    ]:
        (output_dir / name).unlink(missing_ok=True)


def remove_experiment_outputs(output_dir: Path) -> None:
    for name in [
        "train.csv",
        "test.json",
        "best.pt",
        "loss_curves.png",
        "acc_curves.png",
        "confusion_matrix.png",
    ]:
        (output_dir / name).unlink(missing_ok=True)


def build_vgg(model_cfg: dict):
    models = {
        "vgg_a": VGG_A,
        "vgg_a_batchnorm": VGG_A_BatchNorm,
    }
    return models[model_cfg["name"]](dropout=float(model_cfg.get("dropout", 0.0)))


def main() -> None:
    args = parse_args()
    configs = load_selected_configs(CONFIG_DIR, EXPERIMENT_ORDER, args.config)
    should_plot_figures = args.config is None
    first_cfg = configs[0]
    set_seed(int(first_cfg.get("seed", 42)))
    device = get_device()

    output_dir = ROOT / "outputs" / "bn"
    output_dir.mkdir(parents=True, exist_ok=True)
    if should_plot_figures:
        remove_root_outputs(output_dir)

    landscape_runs: list[dict] = []
    analysis_runs: list[dict] = []
    for cfg in configs:
        exp_id = cfg["id"]
        exp_dir = output_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        remove_experiment_outputs(exp_dir)

        data_cfg = cfg["data"]
        model_cfg = cfg["model"]
        train_loader, test_loader = build_cifar10_train_test_loaders(
            data_dir=resolve_path(data_cfg["data_dir"]),
            batch_size=int(data_cfg["batch_size"]),
            num_workers=int(data_cfg["num_workers"]),
            seed=int(cfg.get("seed", 42)),
            mean=data_cfg["mean"],
            std=data_cfg["std"],
        )
        model = build_vgg(model_cfg)
        fit_step_experiment(
            experiment_id=exp_id,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            cfg={
                "model": model_cfg,
                "train": cfg["train"],
                "analysis": cfg.get("analysis", {}),
                "output": cfg.get("output", {}),
            },
            train_csv=exp_dir / "train.csv",
            test_json=exp_dir / "test.json",
            checkpoint_path=exp_dir / "best.pt",
            device=device,
            task="bn",
            class_names=CIFAR10_CLASSES,
        )
        landscape_runs.append(
            {
                "train_csv": exp_dir / "train.csv",
                "model_name": model_cfg["name"],
                "lr": cfg["train"]["lr"],
            }
        )
        if bool(cfg.get("analysis", {}).get("enabled", False)):
            analysis_runs.append(
                {
                    "train_csv": exp_dir / "train.csv",
                    "model_name": model_cfg["name"],
                }
            )

    if should_plot_figures:
        plot_bn_analysis_figures(
            landscape_runs,
            analysis_runs,
            output_dir,
            sample_every_steps=24,
        )


if __name__ == "__main__":
    main()
