from __future__ import annotations

from torch import nn


def build_criterion(loss_name: str, label_smoothing: float = 0.0) -> nn.Module:
    loss_name = loss_name.lower()
    if loss_name != "cross_entropy":
        raise ValueError(f"Unsupported loss: {loss_name}")
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

