import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "framework",
        type=str,
        choices=["torch", "ort"],
        help="The LightGlue framework to measure inference time. Options are 'torch' for PyTorch and 'ort' for ONNXRuntime.",
    )
    parser.add_argument(
        "--megadepth_path",
        type=Path,
        default=Path("megadepth_test_1500"),
        required=False,
        help="Path to the root of the MegaDepth dataset.",
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
        "--fp16",
        action="store_true",
        help="Whether to enable mixed precision for PyTorch, or half-precision for ONNXRuntime.",
    )

    # PyTorch-specific args
    parser.add_argument(
        "--num_keypoints",
        type=int,
        default=1024,
        required=False,
        help="Number of keypoints outputted by the detector. This number must be smaller than image height * width. Defaults to 1024.",
    )

    # ONNXRuntime-specific args
    parser.add_argument(
        "--onnx_path",
        type=str,
        default=None,
        required=False,
        help="Path to ONNX model (end2end).",
    )
    parser.add_argument(
        "--trt",
        action="store_true",
        help="Whether to use TensorRT Execution Provider.",
    )
    return parser.parse_args()


def get_megadepth_images(path: Path):
    sort_key = lambda p: int(p.stem.split("_")[0])
    images = sorted(
        list((path / "Undistorted_SfM/0015/images").glob("*.jpg")), key=sort_key
    ) + sorted(list((path / "Undistorted_SfM/0022/images").glob("*.jpg")), key=sort_key)
    return images


def create_models(
    framework: str, fp16=False, num_keypoints=1024, onnx_path=None, trt=False
):
    if framework == "torch":
        device = torch.device("cuda")
        detector = DeDoDeDetectorL(num_keypoints=num_keypoints).eval().to(device)
        descriptor = DeDoDeDescriptorB().eval().to(device)
        matcher = (
            DualSoftMaxMatcher(normalize=True, inv_temp=20, threshold=0.1)
            .eval()
            .to(device)
        )
        model = DeDoDeEnd2End(detector, descriptor, matcher)
    elif framework == "ort":
        if onnx_path is None:
            onnx_path = (
                f"weights/dedode_end2end_{num_keypoints}"
                f"{'_fp16' if fp16 and not trt else ''}"
                ".onnx"
            )

        providers = ["CUDAExecutionProvider"]  # , "CPUExecutionProvider"]

        if trt:
            providers.insert(
                0,
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_fp16_enable": fp16,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": "weights/cache",
                        "trt_builder_optimization_level": 5,
                    },
                ),
            )
        model = ort.InferenceSession(onnx_path, providers=providers)

    return model


def get_inputs(framework: str, im_A_path, im_B_path, img_size, fp16, trt):
    H, W = img_size

    if framework == "torch":
        im_A = load_image(im_A_path, H, W)
        im_B = load_image(im_B_path, H, W)
        images = torch.concat([im_A, im_B]).cuda()
    elif framework == "ort":
        im_A, im_B = Image.open(im_A_path), Image.open(im_B_path)
        images = DeDoDeRunner.preprocess(
            np.stack([im_A.resize((W, H)), im_B.resize((W, H))])
        )
        if fp16 and not trt:
            images = images.astype(np.float16)

    return images


def measure_inference(framework: str, model, images, fp16) -> float:
    if framework == "torch":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.inference_mode(), torch.autocast("cuda", enabled=fp16):
            result = model(images)
        end.record()
        torch.cuda.synchronize()

        return start.elapsed_time(end)
    elif framework == "ort":
        model_inputs = {"images": images}
        model_outputs = ["matches_A", "matches_B", "batch_ids"]

        # Prepare IO-Bindings
        binding = model.io_binding()

        for name, arr in model_inputs.items():
            binding.bind_cpu_input(name, arr)

        for name in model_outputs:
            binding.bind_output(name, "cuda")

        # Measure only matching time
        start = time.perf_counter()
        result = model.run_with_iobinding(binding)
        end = time.perf_counter()

        return (end - start) * 1000


def evaluate(
    framework: str,
    megadepth_path=Path("megadepth_test_1500"),
    img_size=[256, 256],
    fp16=False,
    num_keypoints=1024,
    onnx_path=None,
    trt=False,
):
    images = get_megadepth_images(megadepth_path)
    image_pairs = list(zip(images[::2], images[1::2]))

    model = create_models(
        framework,
        fp16=fp16,
        num_keypoints=num_keypoints,
        onnx_path=onnx_path,
        trt=trt,
    )

    # Warmup
    for im_A_path, im_B_path in image_pairs[:10]:
        images = get_inputs(
            framework, im_A_path, im_B_path, img_size=img_size, fp16=fp16, trt=trt
        )
        _ = measure_inference(framework, model, images, fp16=fp16)

    # Measure
    timings = []
    for im_A_path, im_B_path in tqdm(image_pairs):
        images = get_inputs(
            framework, im_A_path, im_B_path, img_size=img_size, fp16=fp16, trt=trt
        )

        inference_time = measure_inference(framework, model, images, fp16=fp16)
        timings.append(inference_time)

    # Results
    timings = np.array(timings)
    print(f"Mean inference time: {timings.mean():.2f} +/- {timings.std():.2f} ms")
    print(f"Median inference time: {np.median(timings):.2f} ms")


if __name__ == "__main__":
    args = parse_args()
    if args.framework == "torch":
        import torch

        from DeDoDe import (
            DeDoDeDescriptorB,
            DeDoDeDetectorL,
            DeDoDeEnd2End,
            DualSoftMaxMatcher,
        )
        from DeDoDe.utils import load_image
    elif args.framework == "ort":
        import onnxruntime as ort

        from onnx_runner import DeDoDeRunner

    evaluate(**vars(args))
