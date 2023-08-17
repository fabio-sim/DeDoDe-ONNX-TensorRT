import torch
import torch.nn as nn

from ..utils import to_normalized_coords, to_pixel_coords


def dual_softmax_matcher(
    desc_A: torch.Tensor,  # (B, M, C)
    desc_B: torch.Tensor,  # (B, N, C)
    normalize=False,
    inv_temperature=1,
):
    if normalize:
        desc_A = desc_A / desc_A.norm(dim=-1, keepdim=True)
        desc_B = desc_B / desc_B.norm(dim=-1, keepdim=True)
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    else:
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    P = corr.softmax(dim=-2) * corr.softmax(dim=-1)
    return P


class DualSoftMaxMatcher(nn.Module):
    def __init__(self, normalize=False, inv_temp=1, threshold=0.0):
        super().__init__()
        self.normalize = normalize
        self.inv_temp = inv_temp
        self.threshold = threshold

    def forward(
        self,
        keypoints_A: torch.Tensor,
        descriptions_A: torch.Tensor,
        keypoints_B: torch.Tensor,
        descriptions_B: torch.Tensor,
    ):
        P = dual_softmax_matcher(
            descriptions_A,
            descriptions_B,
            normalize=self.normalize,
            inv_temperature=self.inv_temp,
        )
        ids = torch.nonzero(
            (P == P.max(dim=-1, keepdim=True).values)
            * (P == P.max(dim=-2, keepdim=True).values)
            * (P > self.threshold)
        )
        batch_ids = ids[:, 0]
        matches_A = keypoints_A[batch_ids, ids[:, 1]]
        matches_B = keypoints_B[batch_ids, ids[:, 2]]
        return matches_A, matches_B, batch_ids

    def to_pixel_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_pixel_coords(x_A, H_A, W_A), to_pixel_coords(x_B, H_B, W_B)

    def to_normalized_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_normalized_coords(x_A, H_A, W_A), to_normalized_coords(x_B, H_B, W_B)
