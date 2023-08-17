import argparse

import numpy as np
from PIL import Image

from onnx_runner import DeDoDeRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_paths",
        nargs=2,
        type=str,
        default=["assets/im_A.jpg", "assets/im_B.jpg"],
        required=False,
        help="Paths to two input images for inference.",
    )
    parser.add_argument(
        "--img_size",
        nargs=2,
        type=int,
        default=[256, 256],
        required=False,
        help="Image size for inference. Please provide two integers (height width). Ensure that you have enough memory.",
    )
    parser.add_argument(
        "--detector_path",
        type=str,
        default=None,
        required=False,
        help="Path to the detector ONNX model.",
    )
    parser.add_argument(
        "--descriptor_path",
        type=str,
        default=None,
        required=False,
        help="Path to the descriptor ONNX model.",
    )
    parser.add_argument(
        "--matcher_path",
        type=str,
        default=None,
        required=False,
        help="Path to the matcher ONNX model.",
    )
    parser.add_argument(
        "--end2end",
        action="store_true",
        help="Whether to run inference using an end-to-end pipeline instead of individual models.",
    )
    parser.add_argument(
        "--end2end_path",
        type=str,
        default=None,
        required=False,
        help="Path to the end2end DeDoDe ONNX model.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to run inference using float16 (half) ONNX model (CUDA only).",
    )
    parser.add_argument(
        "--trt",
        action="store_true",
        help="Whether to use TensorRT. Note that the end2end ONNX model must NOT be exported with --fp16. TensorRT will perform the conversion instead. Only static input shapes are supported.",
    )
    parser.add_argument(
        "--viz", action="store_true", help="Whether to visualize the results."
    )
    return parser.parse_args()


def infer(
    img_paths=["assets/im_A.jpg", "assets/im_B.jpg"],
    img_size=[256, 256],
    detector_path=None,
    descriptor_path=None,
    matcher_path=None,
    end2end_path=None,
    end2end=False,
    fp16=False,
    trt=False,
    viz=False,
):
    im_A_path, im_B_path = img_paths
    im_A, im_B = Image.open(im_A_path), Image.open(im_B_path)
    W_A, H_A = im_A.size
    W_B, H_B = im_B.size

    # Handle args.
    if end2end:
        assert (
            detector_path is None and descriptor_path is None and matcher_path is None
        ), "Individual models not used in end2end inference."
        if end2end_path is None:
            end2end_path = "weights/dedode_end2end_3840_trt.onnx"  # default path
    else:
        assert end2end_path is None, "Inference using individual models."
        if detector_path is None:
            detector_path = "weights/detector_L_1024.onnx"  # default path
        if descriptor_path is None:
            descriptor_path = "weights/descriptor_B_1024.onnx"  # default path
        if matcher_path is None:
            matcher_path = "weights/matcher.onnx"  # default path

    # Preprocessing
    H, W = img_size
    images = DeDoDeRunner.preprocess(
        np.stack([im_A.resize((W, H)), im_B.resize((W, H))])
    )
    if fp16 and not trt:
        images = images.astype(np.float16)

    # Inference
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if trt:
        assert end2end, "TensorRT performs optimally when end2end."
        providers.insert(
            0,
            (
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": fp16,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "weights/cache",
                },
            ),
        )

    runner = DeDoDeRunner(
        detector_path, descriptor_path, matcher_path, end2end_path, providers=providers
    )
    matches_A, matches_B, batch_ids = runner.run(images)

    # Postprocessing
    matches_A = DeDoDeRunner.postprocess(matches_A, H_A, W_A)
    matches_B = DeDoDeRunner.postprocess(matches_B, H_B, W_B)

    # Visualisation
    if viz:
        import cv2

        from onnx_runner.viz_utils import draw_matches

        matches = draw_matches(im_A, matches_A, im_B, matches_B)
        cv2.imshow("matches", cv2.cvtColor(matches, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

    return matches_A, matches_B


if __name__ == "__main__":
    args = parse_args()
    matches_A, matches_B = infer(**vars(args))
    print(matches_A, matches_B)
    print(matches_A.shape, matches_B.shape)
