import torch
from torch import nn

from .descriptors.dedode_descriptor import DeDoDeDescriptor
from .detectors.dedode_detector import DeDoDeDetector
from .matchers.dual_softmax_matcher import DualSoftMaxMatcher


class DeDoDeEnd2End(nn.Module):
    def __init__(
        self,
        detector: DeDoDeDetector,
        descriptor: DeDoDeDescriptor,
        matcher: DualSoftMaxMatcher,
    ):
        super().__init__()
        self.detector = detector
        self.descriptor = descriptor
        self.matcher = matcher

    def forward(
        self,
        images: torch.Tensor,  # (2B, C, H, W)
    ):
        keypoints = self.detector(images)
        descriptions = self.descriptor(images, keypoints)

        matches_A, matches_B, batch_ids = self.matcher(
            keypoints[0::2], descriptions[0::2], keypoints[1::2], descriptions[1::2]
        )

        return (
            matches_A,  # (N, 2)
            matches_B,  # (N, 2)
            batch_ids,  # (N,)
        )
