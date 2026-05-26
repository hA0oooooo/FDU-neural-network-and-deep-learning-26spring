from __future__ import annotations

from torch import nn


def get_activation(name: str) -> nn.Module:
    """Return an activation layer by name."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name in {"silu", "swish", "swilu"}:
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ConvBlock(nn.Module):
    """Conv2d -> optional BatchNorm2d -> Activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "relu",
        use_batchnorm: bool = True,
        stride: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=not use_batchnorm,
            )
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(get_activation(activation))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class PlainBlock(nn.Module):
    """Two-convolution CIFAR block without a residual shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "relu",
        use_batchnorm: bool = True,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = ConvBlock(
            in_channels,
            out_channels,
            activation,
            use_batchnorm,
            stride=stride,
        )

        layers: list[nn.Module] = [
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=not use_batchnorm,
            )
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        self.conv2 = nn.Sequential(*layers)
        self.activation = get_activation(activation)

    def forward(self, x):
        return self.activation(self.conv2(self.conv1(x)))


class ResidualBlock(nn.Module):
    """Two-convolution CIFAR residual block with optional projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "relu",
        use_batchnorm: bool = True,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = ConvBlock(
            in_channels,
            out_channels,
            activation,
            use_batchnorm,
            stride=stride,
        )

        layers: list[nn.Module] = [
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=not use_batchnorm,
            )
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        self.conv2 = nn.Sequential(*layers)

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            shortcut_layers: list[nn.Module] = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=not use_batchnorm,
                )
            ]
            if use_batchnorm:
                shortcut_layers.append(nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(*shortcut_layers)

        self.activation = get_activation(activation)

    def forward(self, x):
        return self.activation(self.conv2(self.conv1(x)) + self.shortcut(x))
