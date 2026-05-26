from __future__ import annotations

from torch import nn


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def make_vgg_layers(use_batchnorm: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = 3
    cfg: list[int | str] = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    for item in cfg:
        if item == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        out_channels = int(item)
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=not use_batchnorm,
            )
        )
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels

    return nn.Sequential(*layers)


class _VGGA(nn.Module):
    def __init__(
        self,
        use_batchnorm: bool,
        num_classes: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.features = make_vgg_layers(use_batchnorm)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self.apply(_init_weights)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class VGG_A(_VGGA):
    """CIFAR-10 VGG-A without BatchNorm."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.0) -> None:
        super().__init__(use_batchnorm=False, num_classes=num_classes, dropout=dropout)


class VGG_A_BatchNorm(_VGGA):
    """CIFAR-10 VGG-A with Conv2d -> BatchNorm2d -> ReLU blocks."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.0) -> None:
        super().__init__(use_batchnorm=True, num_classes=num_classes, dropout=dropout)
