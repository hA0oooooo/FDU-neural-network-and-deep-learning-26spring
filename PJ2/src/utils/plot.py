from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import read_metrics

BN_MODEL_LABELS = {
    "vgg_a": "Standard VGG",
    "vgg_a_batchnorm": "Standard VGG + BatchNorm",
}
BN_COLORS = {
    "vgg_a": "#55a868",
    "vgg_a_batchnorm": "#c44e52",
}
COMPARISON_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
COMPARISON_LABELS = {
    "cls_baseline": "Baseline",
    "cls_large": "Large",
    "cls_ultra": "Ultra",
    "cls_activation_silu": "SiLU",
    "cls_optimizer_sgd": "SGD + Momentum",
    "cls_loss_label_smoothing": "Label Smoothing",
    "cls_cutmix": "CutMix",
    "cls_best": "Best",
    "cls_best_sam": "Best + SAM",
}


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_train_curves(
    train_csv: str | Path,
    loss_path: str | Path,
    acc_path: str | Path,
    title: str,
) -> None:
    df = read_metrics(train_csv).sort_values("epoch")
    for column in ["train_loss", "val_loss", "train_acc", "val_acc"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    _ensure_parent(loss_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["train_loss"], label="train")
    ax.plot(df["epoch"], df["val_loss"], label="val")
    ax.set_title(f"{title} Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_path, dpi=320)
    plt.close(fig)

    _ensure_parent(acc_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["train_acc"], label="train")
    ax.plot(df["epoch"], df["val_acc"], label="val")
    ax.set_title(f"{title} Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(acc_path, dpi=320)
    plt.close(fig)


def plot_train_comparison(
    experiment_train_csvs: dict[str, str | Path],
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    loss_path = save_path.with_name(f"{save_path.stem}_loss{save_path.suffix}")
    acc_path = save_path.with_name(f"{save_path.stem}_accuracy{save_path.suffix}")
    _ensure_parent(loss_path)
    loss_fig, loss_ax = plt.subplots(figsize=(6.4, 4.0))
    acc_fig, acc_ax = plt.subplots(figsize=(6.4, 4.0))

    for index, (exp_id, train_csv) in enumerate(experiment_train_csvs.items()):
        df = read_metrics(train_csv).sort_values("epoch")
        for column in ["train_loss", "val_loss", "train_acc", "val_acc"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        label = COMPARISON_LABELS.get(exp_id, exp_id.replace("_", " "))
        color = COMPARISON_COLORS[index % len(COMPARISON_COLORS)]
        loss_ax.plot(
            df["epoch"],
            df["train_loss"],
            color=color,
            linewidth=0.7,
            linestyle="--",
            label=f"{label} train",
        )
        loss_ax.plot(
            df["epoch"], df["val_loss"], color=color, linewidth=0.85, label=f"{label} val"
        )
        acc_ax.plot(
            df["epoch"],
            df["train_acc"],
            color=color,
            linewidth=0.7,
            linestyle="--",
            label=f"{label} train",
        )
        acc_ax.plot(
            df["epoch"], df["val_acc"], color=color, linewidth=0.85, label=f"{label} val"
        )

    for ax in (loss_ax, acc_ax):
        ax.grid(True, linestyle="--", linewidth=0.45, color="#d7dce2", alpha=0.8)
        ax.tick_params(axis="both", labelsize=8)
        ax.legend(fontsize=7, framealpha=0.92)

    loss_ax.set_title("Loss Curves", fontsize=11)
    loss_ax.set_xlabel("Epoch", fontsize=9)
    loss_ax.set_ylabel("Loss", fontsize=9)

    acc_ax.set_title("Accuracy Curves", fontsize=11)
    acc_ax.set_xlabel("Epoch", fontsize=9)
    acc_ax.set_ylabel("Accuracy (%)", fontsize=9)

    loss_fig.tight_layout()
    loss_fig.savefig(loss_path, dpi=360)
    plt.close(loss_fig)
    acc_fig.tight_layout()
    acc_fig.savefig(acc_path, dpi=360)
    plt.close(acc_fig)


def plot_confusion_matrix_data(
    matrix: list[list[int]],
    class_names: list[str],
    save_path: str | Path,
    title: str = "CIFAR-10 Confusion Matrix",
) -> None:
    matrix_array = np.asarray(matrix)
    _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix_array, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = matrix_array.max() / 2.0
    for i in range(matrix_array.shape[0]):
        for j in range(matrix_array.shape[1]):
            ax.text(
                j,
                i,
                str(matrix_array[i, j]),
                ha="center",
                va="center",
                color="white" if matrix_array[i, j] > threshold else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(save_path, dpi=320)
    plt.close(fig)


def plot_confusion_from_test_json(test_json: str | Path, save_path: str | Path) -> None:
    with Path(test_json).open("r", encoding="utf-8") as f:
        result = json.load(f)
    plot_confusion_matrix_data(
        result["confusion_matrix"],
        result["classes"],
        save_path,
        title=f"{COMPARISON_LABELS.get(result['experiment_id'], result['experiment_id'])} Confusion Matrix",
    )


def _plot_loss_landscape_frame(
    df: pd.DataFrame,
    save_path: str | Path,
) -> None:
    df["train_loss"] = pd.to_numeric(df["train_loss"], errors="coerce")
    df["global_step"] = pd.to_numeric(df["global_step"], errors="coerce")
    df["lr"] = pd.to_numeric(df["lr"], errors="coerce")

    _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor("#EAEAF2")
    fig.patch.set_facecolor("white")
    for model_name, label in BN_MODEL_LABELS.items():
        sub = df[df["model_name"] == model_name].copy()
        if sub.empty:
            continue

        envelope = (
            sub.dropna(subset=["global_step", "train_loss"])
            .groupby("global_step")["train_loss"]
            .agg(["min", "max"])
            .reset_index()
            .sort_values("global_step")
        )
        if envelope.empty:
            continue

        x = envelope["global_step"].to_numpy(dtype=float)
        lower = envelope["min"].to_numpy(dtype=float)
        upper = envelope["max"].to_numpy(dtype=float)
        color = BN_COLORS[model_name]
        ax.fill_between(x, lower, upper, color=color, alpha=0.35, label=label)
        ax.plot(x, lower, color=color, linewidth=1.2)
        ax.plot(x, upper, color=color, linewidth=1.2)

    ax.set_title("Loss Landscape", fontsize=18)
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel(r"$L(w)$", fontsize=12)
    ax.grid(True, color="white", linewidth=1.0)
    ax.legend(loc="upper right", fontsize=12, frameon=False)
    ax.margins(x=0.01, y=0.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_loss_landscape_from_train_csvs(
    train_csvs: list[str | Path | dict],
    save_path: str | Path,
    sample_every_steps: int = 24,
) -> None:
    frames = []
    for item in train_csvs:
        if isinstance(item, dict):
            frame = read_metrics(item["train_csv"])
            frame["model_name"] = item["model_name"]
            frame["lr"] = item["lr"]
        else:
            frame = read_metrics(item)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["global_step"] = pd.to_numeric(df["global_step"], errors="coerce")
    df = df[df["global_step"] % sample_every_steps == 0]
    _plot_loss_landscape_frame(df, save_path)


def _read_run_frames(runs: list[dict]) -> pd.DataFrame:
    frames = []
    for item in runs:
        frame = read_metrics(item["train_csv"])
        frame["model_name"] = item["model_name"]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["global_step"] = pd.to_numeric(df["global_step"], errors="coerce")
    return df.sort_values("global_step")


def _style_bn_axis(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor("#EAEAF2")
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, color="white", linewidth=1.0)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=12, frameon=False)
    ax.margins(x=0.01, y=0.05)


def plot_bn_metric_envelope(
    runs: list[dict],
    save_path: str | Path,
    min_column: str,
    max_column: str,
    title: str,
    ylabel: str,
) -> None:
    df = _read_run_frames(runs)
    _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    for model_name, label in BN_MODEL_LABELS.items():
        sub = df[df["model_name"] == model_name].copy()
        if sub.empty:
            continue
        sub[min_column] = pd.to_numeric(sub[min_column], errors="coerce")
        sub[max_column] = pd.to_numeric(sub[max_column], errors="coerce")
        sub = sub.dropna(subset=["global_step", min_column, max_column])
        if sub.empty:
            continue
        x = sub["global_step"].to_numpy(dtype=float)
        lower = sub[min_column].to_numpy(dtype=float)
        upper = sub[max_column].to_numpy(dtype=float)
        color = BN_COLORS[model_name]
        ax.fill_between(x, lower, upper, color=color, alpha=0.35, label=label)
        ax.plot(x, upper, color=color, linewidth=1.2)
    _style_bn_axis(ax, title, ylabel)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_bn_metric_lines(
    runs: list[dict],
    save_path: str | Path,
    column: str,
    title: str,
    ylabel: str,
) -> None:
    df = _read_run_frames(runs)
    _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    for model_name, label in BN_MODEL_LABELS.items():
        sub = df[df["model_name"] == model_name].copy()
        if sub.empty:
            continue
        sub[column] = pd.to_numeric(sub[column], errors="coerce")
        sub = sub.dropna(subset=["global_step", column])
        if sub.empty:
            continue
        ax.plot(
            sub["global_step"],
            sub[column],
            color=BN_COLORS[model_name],
            linewidth=1.2,
            label=label,
        )
    _style_bn_axis(ax, title, ylabel)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_bn_analysis_figures(
    landscape_runs: list[dict],
    analysis_runs: list[dict],
    output_dir: str | Path,
    sample_every_steps: int = 24,
) -> None:
    output_dir = Path(output_dir)
    plot_loss_landscape_from_train_csvs(
        landscape_runs,
        output_dir / "loss_landscape.png",
        sample_every_steps=sample_every_steps,
    )
    plot_bn_metric_envelope(
        analysis_runs,
        output_dir / "gradient_predictiveness.png",
        min_column="grad_predict_min",
        max_column="grad_predict_max",
        title="Gradient Predictiveness",
        ylabel=r"$\|\nabla L(w+\rho d)-\nabla L(w)\|_2$",
    )
    plot_bn_metric_envelope(
        analysis_runs,
        output_dir / "effective_beta_smoothness.png",
        min_column="beta_smooth_min",
        max_column="beta_smooth_max",
        title="Effective β-smoothness",
        ylabel=r"$\beta_{\mathrm{eff}}(\rho)=\|\nabla L(w+\rho d)-\nabla L(w)\|_2/\rho$",
    )
    plot_bn_metric_lines(
        analysis_runs,
        output_dir / "sam_sharpness.png",
        column="sam_sharpness",
        title="SAM-style Sharpness",
        ylabel=r"$\widehat{S}_\rho(w)=L(w+\hat{\epsilon})-L(w)$",
    )
