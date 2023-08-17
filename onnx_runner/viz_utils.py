import cv2
import numpy as np


def draw_matches(im_A, kpts_A, im_B, kpts_B):
    kpts_A = [cv2.KeyPoint(x, y, 1.0) for x, y in kpts_A]
    kpts_B = [cv2.KeyPoint(x, y, 1.0) for x, y in kpts_B]
    matches_A_to_B = [cv2.DMatch(idx, idx, 0.0) for idx in range(len(kpts_A))]
    im_A, im_B = np.array(im_A), np.array(im_B)
    ret = cv2.drawMatches(im_A, kpts_A, im_B, kpts_B, matches_A_to_B, None)
    return ret
