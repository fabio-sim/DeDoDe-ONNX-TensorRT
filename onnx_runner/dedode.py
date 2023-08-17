# No dependency on PyTorch

import numpy as np
import onnxruntime as ort

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DeDoDeRunner:
    def __init__(
        self,
        detector_path=None,
        descriptor_path=None,
        matcher_path=None,
        end2end_path=None,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    ):
        if end2end_path is not None:
            assert (
                detector_path is None
                and descriptor_path is None
                and matcher_path is None
            ), "Inference using end2end model does not require individual models."

            self.is_end2end = True
            self.end2end = ort.InferenceSession(end2end_path, providers=providers)
        else:
            assert (
                detector_path is not None
                and descriptor_path is not None
                and matcher_path is not None
            ), "All three model components must be provided if not using end2end."

            self.is_end2end = False
            self.detector = ort.InferenceSession(detector_path, providers=providers)
            self.descriptor = ort.InferenceSession(descriptor_path, providers=providers)
            self.matcher = ort.InferenceSession(matcher_path, providers=providers)

    def run(self, images: np.ndarray):
        if self.is_end2end:
            matches_A, matches_B, batch_ids = self.end2end.run(None, {"images": images})
        else:
            image_A = images[0:1]
            image_B = images[1:2]
            keypoints_A = self.detector.run(None, {"image": image_A})[0]
            keypoints_B = self.detector.run(None, {"image": image_B})[0]
            descriptions_A = self.descriptor.run(
                None, {"image": image_A, "keypoints": keypoints_A}
            )[0]
            descriptions_B = self.descriptor.run(
                None, {"image": image_B, "keypoints": keypoints_B}
            )[0]
            matches_A, matches_B, batch_ids = self.matcher.run(
                None,
                {
                    "keypoints_A": keypoints_A,
                    "description_A": descriptions_A,
                    "keypoints_B": keypoints_B,
                    "description_B": descriptions_B,
                },
            )

        return matches_A, matches_B, batch_ids

    @staticmethod
    def preprocess(images: np.ndarray) -> np.ndarray:
        # images.shape == (B, H, W, C)
        images = (images / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return images.transpose(0, 3, 1, 2).astype(np.float32)

    @staticmethod
    def postprocess(matches: np.ndarray, H: int, W: int) -> np.ndarray:
        return (matches + 1) / 2 * [W, H]
