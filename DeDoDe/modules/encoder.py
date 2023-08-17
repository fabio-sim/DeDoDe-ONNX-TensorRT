import torch
import torch.nn as nn
import torchvision.models as tvm


class VGG(nn.Module):
    def __init__(self, size: str = "19", weights=None):
        super().__init__()
        if size == "11":
            self.layers = nn.ModuleList(tvm.vgg11_bn(weights=weights).features[:22])
        elif size == "13":
            self.layers = nn.ModuleList(tvm.vgg13_bn(weights=weights).features[:28])
        elif size == "19":
            self.layers = nn.ModuleList(tvm.vgg19_bn(weights=weights).features[:40])
        else:
            raise ValueError(
                f"Unsupported VGG size: {size}. Size (str) must be one of: '11', '13', '19'."
            )
        # Maxpool layers: 6, 13, 26, 39

    def forward(self, x: torch.Tensor):
        feats = []
        sizes = []
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                feats.append(x)
                sizes.append(x.shape[-2:])
            x = layer(x)
        return feats, sizes
