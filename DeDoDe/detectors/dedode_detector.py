import torch
import torch.nn as nn
import torch.nn.functional as F


class DeDoDeDetector(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, num_keypoints=10000):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_keypoints = num_keypoints

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features, sizes = self.encoder(images)
        logits: torch.Tensor = 0
        context: torch.Tensor = None
        scales = ["8", "4", "2", "1"]
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales)):
            delta_logits, context = self.decoder(
                feature_map, context=context, scale=scale
            )
            logits = logits + delta_logits
            if idx < len(scales) - 1:
                size = sizes[-(idx + 2)]
                logits = F.interpolate(
                    logits, size=size, mode="bicubic", align_corners=False
                )
                context = F.interpolate(
                    context, size=size, mode="bilinear", align_corners=False
                )

        B, K, H, W = logits.shape
        keypoint_p = (
            logits.reshape(B, K * H * W).softmax(dim=-1).reshape(B, K, H * W).sum(dim=1)
        )
        keypoints = sample_keypoints(
            keypoint_p.reshape(B, H, W),
            use_nms=False,
            sample_topk=True,
            num_samples=self.num_keypoints,
            return_scoremap=False,
            sharpen=False,
            upsample=False,
            increase_coverage=True,
        )
        return keypoints


def get_grid(B, H, W, device):
    x1_n = torch.meshgrid(
        *[torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device) for n in (B, H, W)],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
    return x1_n


def sample_keypoints(
    scoremap: torch.Tensor,
    num_samples=8192,
    use_nms=True,
    sample_topk=False,
    return_scoremap=False,
    sharpen=False,
    upsample=False,
    increase_coverage=False,
):
    device = scoremap.device

    # scoremap = scoremap**2
    log_scoremap = (scoremap + 1e-10).log()
    if upsample:
        log_scoremap = F.interpolate(
            log_scoremap[:, None], scale_factor=3, mode="bicubic", align_corners=False
        )[
            :, 0
        ]  # .clamp(min = 0)
        scoremap = log_scoremap.exp()

    B, H, W = scoremap.shape
    if increase_coverage:
        weights = (-torch.linspace(-2, 2, steps=51, device=device) ** 2).exp()[
            None, None
        ]
        # 10000 is just some number for maybe numerical stability, who knows. :), result is invariant anyway
        local_density_x = F.conv2d(
            (scoremap[:, None] + 1e-6) * 10000,
            weights[..., None, :],
            padding=(0, 51 // 2),
        )
        local_density = F.conv2d(
            local_density_x, weights[..., None], padding=(51 // 2, 0)
        )[:, 0]
        scoremap = scoremap * (local_density + 1e-8) ** (-1 / 2)

    if sharpen:
        laplace_operator = (
            torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], device=device) / 4
        )
        scoremap = scoremap[:, None] - 0.5 * F.conv2d(
            scoremap[:, None], weight=laplace_operator, padding=1
        )
        scoremap = scoremap[:, 0].clamp(min=0)

    if use_nms:
        scoremap = scoremap * (
            scoremap == F.max_pool2d(scoremap, (3, 3), stride=1, padding=1)
        )

    if sample_topk:
        inds = torch.topk(scoremap.reshape(B, H * W), k=num_samples).indices
    else:
        inds = torch.multinomial(
            scoremap.reshape(B, H * W), num_samples=num_samples, replacement=False
        )

    grid = get_grid(B, H, W, device=device).reshape(B, H * W, 2)
    kps = torch.gather(grid, dim=1, index=inds[..., None].expand(B, num_samples, 2))

    if return_scoremap:
        return kps, torch.gather(scoremap.reshape(B, H * W), dim=1, index=inds)

    return kps
