import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleDict, num_prototypes=1):
        super().__init__()
        self.layers = layers
        self.scales = self.layers.keys()
        self.num_prototypes = num_prototypes

    def forward(self, features: torch.Tensor, scale: str, context: torch.Tensor = None):
        if context is not None:
            features = torch.cat((features, context), dim=1)
        stuff = self.layers[scale](features)
        logits, context = (
            stuff[:, : self.num_prototypes],
            stuff[:, self.num_prototypes :],
        )
        return logits, context


class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=True,
        kernel_size=5,
        hidden_blocks=5,
        residual=False,
    ):
        super().__init__()
        self.block1 = self.create_block(
            in_dim,
            hidden_dim,
            dw=False,
            kernel_size=1,
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                )
                for _ in range(hidden_blocks)
            ]
        )
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        self.residual = residual

    def create_block(
        self,
        in_dim: int,
        out_dim: int,
        dw=True,
        kernel_size=5,
        bias=True,
        norm_type=nn.BatchNorm2d,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert (
                out_dim % in_dim == 0
            ), "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        norm = (
            norm_type(out_dim)
            if norm_type is nn.BatchNorm2d
            else norm_type(num_channels=out_dim)
        )
        relu = nn.ReLU()
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x0 = self.block1(feats)
        x = self.hidden_blocks(x0)
        if self.residual:
            x = (x + x0) / 1.4
        x = self.out_conv(x)
        return x
