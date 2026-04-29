import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_robustness_bar(csv_path="outputs/test_model.csv", save_path="outputs/figures/robust_bar.png"):
    models = ["mlp", "cnn", "cnn_dataaug"]
    labels = ["MLP", "CNN", "CNN + DataAug"]
    datasets = ["clean", "rotate", "translate", "resize", "transform", "gaussian_0.05", "gaussian_0.10", "gaussian_0.20"]

    scores = {}
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            scores[(row["model"], row["data"])] = float(row["test_accuracy"])

    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (model, label) in enumerate(zip(models, labels)):
        y = [scores[(model, dataset)] for dataset in datasets]
        ax.bar(x + (i - 1) * width, y, width, label=label)

    ax.set_ylabel("test accuracy")
    ax.set_xlabel("perturbation")
    ax.set_title("Accuracy under different perturbations")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_robustness_bar()
