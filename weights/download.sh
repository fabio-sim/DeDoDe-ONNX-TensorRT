#!/bin/bash

RELEASE=v1.0.0
NUM_KEYPOINTS=1024

curl -L https://github.com/fabio-sim/DeDoDe-ONNX-TensorRT/releases/download/${RELEASE}/dedode_end2end_${NUM_KEYPOINTS}.onnx -o weights/dedode_end2end_${NUM_KEYPOINTS}.onnx
curl -L https://github.com/fabio-sim/DeDoDe-ONNX-TensorRT/releases/download/${RELEASE}/dedode_end2end_${NUM_KEYPOINTS}_fp16.onnx -o weights/dedode_end2end_${NUM_KEYPOINTS}_fp16.onnx
curl -L https://github.com/fabio-sim/DeDoDe-ONNX-TensorRT/releases/download/${RELEASE}/dedode_end2end_${NUM_KEYPOINTS}_trt.onnx -o weights/dedode_end2end_${NUM_KEYPOINTS}_trt.onnx

curl -L https://github.com/fabio-sim/DeDoDe-ONNX-TensorRT/releases/download/${RELEASE}/detector_L_${NUM_KEYPOINTS}.onnx -o weights/detector_L_${NUM_KEYPOINTS}.onnx
curl -L https://github.com/fabio-sim/DeDoDe-ONNX-TensorRT/releases/download/${RELEASE}/detector_L_${NUM_KEYPOINTS}_fp16.onnx -o weights/detector_L_${NUM_KEYPOINTS}_fp16.onnx
curl -L https://github.com/fabio-sim/DeDoDe-ONNX-TensorRT/releases/download/${RELEASE}/descriptor_B_${NUM_KEYPOINTS}.onnx -o weights/descriptor_B_${NUM_KEYPOINTS}.onnx
curl -L https://github.com/fabio-sim/DeDoDe-ONNX-TensorRT/releases/download/${RELEASE}/descriptor_B_${NUM_KEYPOINTS}_fp16.onnx -o weights/descriptor_B_${NUM_KEYPOINTS}_fp16.onnx
curl -L https://github.com/fabio-sim/DeDoDe-ONNX-TensorRT/releases/download/${RELEASE}/matcher.onnx -o weights/matcher.onnx
curl -L https://github.com/fabio-sim/DeDoDe-ONNX-TensorRT/releases/download/${RELEASE}/matcher_fp16.onnx -o weights/matcher_fp16.onnx
