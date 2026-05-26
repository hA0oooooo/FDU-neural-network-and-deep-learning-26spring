from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_DIR = ROOT / "configs" / "classification"
EXPERIMENT_ORDER = [
    "cls_baseline",
    "cls_large",
    "cls_ultra",
    "cls_loss_label_smoothing",
    "cls_activation_tanh",
    "cls_activation_silu",
    "cls_optimizer_sgd",
    "cls_no_dropout",
    "cls_no_batchnorm",
    "cls_no_residual",
    "cls_no_normalize",
    "cls_mixup",
    "cls_cutmix",
    "cls_best",
    "cls_best_sam",
]

from src.models.custom_cnn import CustomCNN
from src.utils.config import build_experiment_config, load_selected_configs
from src.utils.data import CIFAR10_CLASSES, build_cifar10_loaders
from src.utils.plot import plot_train_curves
from src.utils.seed import get_device, set_seed
from src.utils.train import fit_supervised_experiment


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+")
    return parser.parse_args()


def remove_outputs(output_dir: Path) -> None:
    for name in [
        "train.csv",
        "test.json",
        "best.pt",
        "loss_curves.png",
        "acc_curves.png",
        "confusion_matrix.png",
    ]:
        (output_dir / name).unlink(missing_ok=True)


def build_custom_cnn(model_cfg: dict) -> CustomCNN:
    return CustomCNN(
        channels=model_cfg["channels"],
        hidden_dim=int(model_cfg["hidden_dim"]),
        activation=model_cfg["activation"],
        use_batchnorm=bool(model_cfg["use_batchnorm"]),
        dropout=float(model_cfg["dropout"]),
        use_residual=bool(model_cfg["use_residual"]),
        blocks_per_stage=model_cfg.get("blocks_per_stage", [2, 2, 2, 2]),
    )


def main() -> None:
    args = parse_args()
    configs = load_selected_configs(CONFIG_DIR, EXPERIMENT_ORDER, args.config)
    first_cfg = configs[0]
    set_seed(int(first_cfg.get("seed", 42)))
    device = get_device()

    output_dir = ROOT / "outputs" / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_outputs(output_dir)

    data_cfg = first_cfg["data"]
    data_dir = resolve_path(data_cfg["data_dir"])
    train_loader, val_loader, test_loader = build_cifar10_loaders(
        data_dir=data_dir,
        batch_size=int(data_cfg["batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
        val_size=int(data_cfg["val_size"]),
        seed=int(first_cfg.get("seed", 42)),
        mean=data_cfg["mean"],
        std=data_cfg["std"],
    )

    for cfg in configs:
        exp_id = cfg["id"]
        exp_dir = output_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        remove_outputs(exp_dir)

        exp_cfg = build_experiment_config(cfg)
        model = build_custom_cnn(exp_cfg["model"])
        fit_supervised_experiment(
            experiment_id=exp_id,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            cfg=exp_cfg,
            train_csv=exp_dir / "train.csv",
            test_json=exp_dir / "test.json",
            checkpoint_path=exp_dir / "best.pt",
            device=device,
            task="classification",
            class_names=CIFAR10_CLASSES,
        )

        plot_train_curves(
            train_csv=exp_dir / "train.csv",
            loss_path=exp_dir / "loss_curves.png",
            acc_path=exp_dir / "acc_curves.png",
            title=exp_id.replace("_", " "),
        )


if __name__ == "__main__":
    main()
