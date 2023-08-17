import torch
import torch.nn as nn
import torch.nn.functional as F


class DeDoDeDescriptor(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        features, sizes = self.encoder(images)
        descriptor: torch.Tensor = 0
        context: torch.Tensor = None
        scales = self.decoder.scales
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales)):
            delta_descriptor, context = self.decoder(
                feature_map, scale=scale, context=context
            )
            descriptor = descriptor + delta_descriptor
            if idx < len(scales) - 1:
                size = sizes[-(idx + 2)]
                descriptor = F.interpolate(
                    descriptor, size=size, mode="bilinear", align_corners=False
                )
                context = F.interpolate(
                    context, size=size, mode="bilinear", align_corners=False
                )
        described_keypoints = F.grid_sample(
            descriptor,
            keypoints[:, None],
            mode="bilinear",
            align_corners=False,
        )[:, :, 0].transpose(-2, -1)
        return described_keypoints
