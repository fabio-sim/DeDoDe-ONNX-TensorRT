[![GitHub](https://img.shields.io/github/license/fabio-sim/DeDoDe-ONNX-TensorRT)](/LICENSE)
[![ONNX](https://img.shields.io/badge/ONNX-grey)](https://onnx.ai/)
[![TensorRT](https://img.shields.io/badge/TensorRT-76B900)](https://developer.nvidia.com/tensorrt)
[![GitHub Repo stars](https://img.shields.io/github/stars/fabio-sim/DeDoDe-ONNX-TensorRT)](https://github.com/fabio-sim/DeDoDe-ONNX-TensorRT/stargazers)
[![GitHub all releases](https://img.shields.io/github/downloads/fabio-sim/DeDoDe-ONNX-TensorRT/total)](https://github.com/fabio-sim/DeDoDe-ONNX-TensorRT/releases)

# DeDoDe-ONNX-TensorRT
Open Neural Network Exchange (ONNX) compatible implementation of [DeDoDe 🎶 Detect, Don't Describe - Describe, Don't Detect, for Local Feature Matching](https://github.com/Parskatt/DeDoDe). Supports TensorRT 🚀.

<p align="center"><img src="assets/matches.jpg" alt="DeDoDe figure" width=80%><br><em>The DeDoDe detector learns to detect 3D consistent repeatable keypoints, which the DeDoDe descriptor learns to match. The result is a powerful decoupled local feature matcher.</em></p>

## 🔥 ONNX Export

Prior to exporting the ONNX models, please install the [requirements](/requirements.txt).

To convert the DeDoDe models to ONNX, run [`export.py`](/export.py). We provide two types of ONNX exports: individual standalone models, and a combined end-to-end pipeline (recommended for convenience) with the `--end2end` flag.

<details>
<summary>Export Example</summary>
<pre>
python export.py \
    --img_size 256 256 \
    --end2end \
    --dynamic_img_size --dynamic_batch \
    --fp16
</pre>
</details>

If you would like to try out inference right away, you can download ONNX models that have already been exported [here](https://github.com/fabio-sim/DeDoDe-ONNX-TensorRT/releases) or run `./weights/download.sh`.

## ⚡ ONNX Inference

With ONNX models in hand, one can perform inference on Python using ONNX Runtime (see [requirements-onnx.txt](/requirements-onnx.txt)).

The DeDoDe inference pipeline has been encapsulated into a runner class:

```python
from onnx_runner import DeDoDeRunner

images = DeDoDeRunner.preprocess(image_array)
# images.shape == (2B, 3, H, W)

# Create ONNXRuntime runner
runner = DeDoDeRunner(
    end2end_path="weights/dedode_end2end_1024.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    # TensorrtExecutionProvider
)

# Run inference
matches_A, matches_B, batch_ids = runner.run(images)

matches_A = DeDoDeRunner.postprocess(matches_A, H_A, W_A)
matches_B = DeDoDeRunner.postprocess(matches_B, H_B, W_B)
```
Alternatively, you can also run [`infer.py`](/infer.py).

<details>
<summary>Inference Example</summary>
<pre>
python infer.py \
    --img_paths assets/im_A.jpg assets/im_B.jpg \
    --img_size 256 256 \
    --end2end \
    --end2end_path weights/dedode_end2end_1024_fp16.onnx \
    --fp16 \
    --viz
</pre>
</details>

## 🚀 TensorRT Support

TensorRT offers the best performance and greatest memory efficiency.

TensorRT inference is supported for the end-to-end model via the TensorRT Execution Provider in ONNXRuntime. Please follow the [official documentation](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) to install TensorRT. The exported ONNX models must undergo [shape inference](/tools/symbolic_shape_infer.py) for compatibility with TensorRT.

<details>
<summary>TensorRT Example</summary>
<pre>
python tools/symbolic_shape_infer.py \
  --input weights/dedode_end2end_1024.onnx \
  --output weights/dedode_end2end_1024_trt.onnx \
  --auto_merge<br>
CUDA_MODULE_LOADING=LAZY && python infer.py \
  --img_paths assets/DSC_0410.JPG assets/DSC_0411.JPG \
  --img_size 256 256 \
  --end2end \
  --end2end_path weights/dedode_end2end_1024_trt.onnx \
  --trt \
  --viz
</pre>
</details>

The first run will take longer because TensorRT needs to initialise the `.engine` and `.profile` files. Subsequent runs should use the cached files. Only static input shapes are supported. Note that TensorRT will rebuild the cache if it encounters a different input shape.

## Inference Time Comparison

(WIP)

## Credits
If you use any ideas from the papers or code in this repo, please consider citing the authors of [DeDoDe](https://arxiv.org/abs/2308.08479). Lastly, if the ONNX or TensorRT versions helped you in any way, please also consider starring this repository.

```txt
@article{edstedt2023dedode,
      title={DeDoDe: Detect, Don't Describe -- Describe, Don't Detect for Local Feature Matching}, 
      author={Johan Edstedt and Georg Bökman and Mårten Wadenbäck and Michael Felsberg},
      year={2023},
      eprint={2308.08479},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
