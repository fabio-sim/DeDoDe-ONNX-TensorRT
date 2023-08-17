import argparse
import warnings

warnings.filterwarnings("ignore", module="onnxconverter_common.float16")

import onnx
import torch
from onnxconverter_common import float16

from DeDoDe import DeDoDeDescriptorB, DeDoDeDetectorL, DeDoDeEnd2End, DualSoftMaxMatcher
from DeDoDe.utils import load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_size",
        nargs=2,
        type=int,
        default=[256, 256],
        required=False,
        help="Sample image size for ONNX tracing. Please provide two integers (height width). Ensure that you have enough memory to run the export.",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="L",
        choices=["L"],
        required=False,
        help="DeDoDe detector variant. Supported detectors are 'L'. Defaults to 'L'.",
    )
    parser.add_argument(
        "--descriptor",
        type=str,
        default="B",
        choices=["B"],
        required=False,
        help="DeDoDe descriptor variant. Supported descriptors are 'B'. Defaults to 'B'.",
    )
    parser.add_argument(
        "--matcher",
        type=str,
        default="dual_softmax",
        choices=["dual_softmax"],
        required=False,
        help="Matcher variant. Supported matchers are 'dual_softmax'. Defaults to 'dual_softmax'.",
    )
    parser.add_argument(
        "--detector_path",
        type=str,
        default=None,
        required=False,
        help="Path to save the detector ONNX model.",
    )
    parser.add_argument(
        "--descriptor_path",
        type=str,
        default=None,
        required=False,
        help="Path to save the descriptor ONNX model.",
    )
    parser.add_argument(
        "--matcher_path",
        type=str,
        default=None,
        required=False,
        help="Path to save the matcher ONNX model.",
    )
    parser.add_argument(
        "--end2end",
        action="store_true",
        help="Whether to export an end-to-end pipeline instead of individual models.",
    )
    parser.add_argument(
        "--end2end_path",
        type=str,
        default=None,
        required=False,
        help="Path to save the end2end DeDoDe ONNX model.",
    )
    parser.add_argument(
        "--dynamic_img_size",
        action="store_true",
        help="Whether to allow dynamic image sizes.",
    )
    parser.add_argument(
        "--dynamic_batch",
        action="store_true",
        help="Whether to allow dynamic batch size.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to also export float16 (half) ONNX model (CUDA only).",
    )

    # Detector-specific args:
    parser.add_argument(
        "--num_keypoints",
        type=int,
        default=1024,
        required=False,
        help="Number of keypoints outputted by the detector. This number must be smaller than image height * width. Defaults to 1024.",
    )

    return parser.parse_args()


def export_onnx(
    img_size=[256, 256],
    im_A_path="assets/im_A.jpg",
    im_B_path="assets/im_B.jpg",
    detector="L",
    descriptor="B",
    matcher="dual_softmax",
    detector_path=None,
    descriptor_path=None,
    matcher_path=None,
    end2end_path=None,
    end2end=False,
    dynamic_img_size=False,
    dynamic_batch=False,
    fp16=False,
    num_keypoints=1024,
):
    # Handle args.
    H, W = img_size
    assert (
        H * W > num_keypoints
    ), "Number of keypoints must be smaller than image height * width."

    if end2end:
        assert (
            detector_path is None and descriptor_path is None and matcher_path is None
        ), "Individual models will be combined in end2end export."
        if end2end_path is None:
            end2end_path = (
                f"weights/dedode_end2end"
                f"{f'_{H}x{W}' if not dynamic_img_size else ''}"
                f"_{num_keypoints}"
                ".onnx"
            )
    else:
        assert end2end_path is None, "Exporting individual models."
        if detector_path is None:
            detector_path = (
                f"weights/detector_{detector}"
                f"{f'_{H}x{W}' if not dynamic_img_size else ''}"
                f"_{num_keypoints}"
                ".onnx"
            )

        if descriptor_path is None:
            descriptor_path = (
                f"weights/descriptor_{descriptor}"
                f"{f'_{H}x{W}' if not dynamic_img_size else ''}"
                f"_{num_keypoints}"
                ".onnx"
            )

        if matcher_path is None:
            matcher_path = f"weights/matcher.onnx"

    # Load inputs and models.
    device = torch.device(
        "cuda"
    )  # Can also export on CPU if you have more free RAM than VRAM. Must export on CUDA for FP16.

    im_A = load_image(im_A_path, H=H, W=W).to(device)
    im_B = load_image(im_B_path, H=H, W=W).to(device)

    detector = DeDoDeDetectorL(num_keypoints=num_keypoints).eval().to(device)
    descriptor = DeDoDeDescriptorB().eval().to(device)
    matcher = (
        DualSoftMaxMatcher(normalize=True, inv_temp=20, threshold=0.1).eval().to(device)
    )
    dedode_end2end = DeDoDeEnd2End(detector, descriptor, matcher)

    # Export.
    opset_version = 16  # Minimum 16 due to grid sample op.
    if end2end:
        images = torch.concat([im_A, im_B])

        dynamic_axes = {
            "images": {},
            "matches_A": {0: "num_matches"},
            "matches_B": {0: "num_matches"},
            "batch_ids": {0: "num_matches"},
        }
        if dynamic_batch:
            dynamic_axes["images"].update({0: "batch_size"})
        if dynamic_img_size:
            dynamic_axes["images"].update({2: "height", 3: "width"})

        torch.onnx.export(
            dedode_end2end,
            images,
            end2end_path,
            input_names=["images"],
            output_names=["matches_A", "matches_B", "batch_ids"],
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
        )

        if fp16:
            convert_fp16(end2end_path)
    else:
        # Prepare intermediate inputs to individual models.
        with torch.no_grad():
            keypoints_A = detector(im_A)
            keypoints_B = detector(im_B)
            description_A = descriptor(im_A, keypoints_A)
            description_B = descriptor(im_B, keypoints_B)

        dynamic_axes = {"image": {}, "keypoints": {}}
        if dynamic_batch:
            dynamic_axes["image"].update({0: "batch_size"})
            dynamic_axes["keypoints"].update({0: "batch_size"})
        if dynamic_img_size:
            dynamic_axes["image"].update({2: "height", 3: "width"})

        torch.onnx.export(
            detector,
            im_A,
            detector_path,
            input_names=["image"],
            output_names=["keypoints"],
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
        )
        if fp16:
            convert_fp16(detector_path)

        dynamic_axes = {"image": {}, "keypoints": {}, "description": {}}
        if dynamic_batch:
            dynamic_axes["image"].update({0: "batch_size"})
            dynamic_axes["keypoints"].update({0: "batch_size"})
            dynamic_axes["description"].update({0: "batch_size"})
        if dynamic_img_size:
            dynamic_axes["image"].update({2: "height", 3: "width"})

        torch.onnx.export(
            descriptor,
            (im_A, keypoints_A),
            descriptor_path,
            input_names=["image", "keypoints"],
            output_names=["description"],
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
        )
        if fp16:
            convert_fp16(descriptor_path)

        dynamic_axes = {
            "keypoints_A": {},
            "description_A": {},
            "keypoints_B": {},
            "description_B": {},
            "matches_A": {0: "num_matches"},
            "matches_B": {0: "num_matches"},
            "batch_ids": {0: "num_matches"},
        }
        if dynamic_batch:
            dynamic_axes["keypoints_A"].update({0: "batch_size"})
            dynamic_axes["description_A"].update({0: "batch_size"})
            dynamic_axes["keypoints_B"].update({0: "batch_size"})
            dynamic_axes["description_B"].update({0: "batch_size"})

        torch.onnx.export(
            matcher,
            (keypoints_A, description_A, keypoints_B, description_B),
            matcher_path,
            input_names=[
                "keypoints_A",
                "description_A",
                "keypoints_B",
                "description_B",
            ],
            output_names=["matches_A", "matches_B", "batch_ids"],
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
        )
        if fp16:
            convert_fp16(matcher_path)


def convert_fp16(onnx_model_path: str):
    end2end_onnx = onnx.load(onnx_model_path)
    end2end_fp16 = float16.convert_float_to_float16(end2end_onnx)
    onnx.save(end2end_fp16, onnx_model_path.replace(".onnx", "_fp16.onnx"))


if __name__ == "__main__":
    args = parse_args()
    export_onnx(**vars(args))
