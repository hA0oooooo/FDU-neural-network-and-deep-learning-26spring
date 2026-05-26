from __future__ import annotations

from torch import nn

from .blocks import ConvBlock, PlainBlock, ResidualBlock, get_activation


class CustomCNN(nn.Module):
    """Configurable CIFAR-10 CNN used for the main classification experiments."""

    def __init__(
        self,
        channels: list[int] | tuple[int, ...],
        hidden_dim: int,
        activation: str,
        use_batchnorm: bool,
        dropout: float,
        use_residual: bool,
        blocks_per_stage: list[int] | tuple[int, ...] = (2, 2, 2, 2),
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("CustomCNN expects exactly four stage channel values.")
        if len(blocks_per_stage) != 4:
            raise ValueError("CustomCNN expects exactly four block counts.")

        self.stem = ConvBlock(
            3,
            channels[0],
            activation=activation,
            use_batchnorm=use_batchnorm,
            stride=1,
        )
        self.features = self._make_stages(
            channels=channels,
            blocks_per_stage=blocks_per_stage,
            activation=activation,
            use_batchnorm=use_batchnorm,
            use_residual=use_residual,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(channels[-1], hidden_dim),
            get_activation(activation),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def _make_stages(
        self,
        channels: list[int] | tuple[int, ...],
        blocks_per_stage: list[int] | tuple[int, ...],
        activation: str,
        use_batchnorm: bool,
        use_residual: bool,
    ) -> nn.Sequential:
        stages: list[nn.Module] = []
        in_channels = channels[0]
        for stage_idx, (out_channels, num_blocks) in enumerate(zip(channels, blocks_per_stage)):
            stride = 1 if stage_idx == 0 else 2
            blocks: list[nn.Module] = []
            for block_idx in range(num_blocks):
                block_stride = stride if block_idx == 0 else 1
                block_cls = ResidualBlock if use_residual else PlainBlock
                blocks.append(
                    block_cls(
                        in_channels,
                        out_channels,
                        activation=activation,
                        use_batchnorm=use_batchnorm,
                        stride=block_stride,
                    )
                )
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        return nn.Sequential(*stages)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
