#!/bin/bash
set -e

ARCH=${ARCH:-$(rocminfo | grep "Name:" | grep "gfx" | awk 'NR==1' | awk '{print $2}')}
SETUP_DIR=$(dirname -- $0)
SCRIPTS_DIR=${SETUP_DIR}/dataset_and_model
BASE_IMAGE_NAME=rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0
MLPERF_IMAGE_NAME=mlperf_inference_submission_model_and_dataset_prep:5.1

if [[ "$ARCH" == "gfx950" ]]; then
    BASE_IMAGE_NAME=rocm/7.0-preview:rocm7.0_preview_ubuntu_22.04_vllm_0.8.5_mi35X_prealpha1
fi

docker build --no-cache --build-arg BASE_IMAGE=${BASE_IMAGE_NAME} --build-arg SCRIPTS_DIR=${SCRIPTS_DIR} -t ${MLPERF_IMAGE_NAME} -f "$SETUP_DIR/Dockerfile.model_and_dataset" .

