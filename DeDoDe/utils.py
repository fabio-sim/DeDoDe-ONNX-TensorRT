import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

IMAGENET_PREPROCESS = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


def load_image(im_path, H=784, W=784, transform=IMAGENET_PREPROCESS) -> torch.Tensor:
    img = Image.open(im_path).resize((W, H))
    img = np.array(img) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    if transform is not None:
        img = transform(img)
    return img.float()[None]


def to_pixel_coords(flow: torch.Tensor, h1, w1) -> torch.Tensor:
    flow = torch.stack(
        (
            w1 * (flow[..., 0] + 1) / 2,
            h1 * (flow[..., 1] + 1) / 2,
        ),
        axis=-1,
    )
    return flow


def to_normalized_coords(flow, h1, w1):
    flow = torch.stack(
        (
            2 * (flow[..., 0]) / w1 - 1,
            2 * (flow[..., 1]) / h1 - 1,
        ),
        axis=-1,
    )
    return flow


def draw_matches(im_A, kpts_A, im_B, kpts_B):
    if isinstance(kpts_A, torch.Tensor):
        kpts_A = kpts_A.cpu().numpy()
    if isinstance(kpts_B, torch.Tensor):
        kpts_B = kpts_B.cpu().numpy()

    kpts_A = [cv2.KeyPoint(x, y, 1.0) for x, y in kpts_A]
    kpts_B = [cv2.KeyPoint(x, y, 1.0) for x, y in kpts_B]
    matches_A_to_B = [cv2.DMatch(idx, idx, 0.0) for idx in range(len(kpts_A))]
    im_A, im_B = np.array(im_A), np.array(im_B)
    ret = cv2.drawMatches(im_A, kpts_A, im_B, kpts_B, matches_A_to_B, None)
    return ret
