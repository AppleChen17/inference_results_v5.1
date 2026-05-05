#!/bin/bash
set -e

SETUP_DIR=$(dirname -- $0)
# ARCH=${ARCH:-$(rocminfo | grep "Name:" | grep "gfx" | awk 'NR==1' | awk '{print $2}')}
ARCH="gfx1201"
VLLM_IMAGE_NAME=rocm/vllm-dev:nightly_0610_rc2_0610_rc2_20250605
PATCHES="--apply-vllm-patches --apply-llama2-patches"
if [[ "$ARCH" == "gfx950" ]]; then
    VLLM_IMAGE_NAME=rocm/7.0-preview:rocm7.0_preview_ubuntu_22.04_vllm_0.8.5_mi35X_prealpha1
    PATCHES="$PATCHES --apply-aiter-patches"
elif [[ "$ARCH" == "gfx942" ]]; then
    PATCHES="$PATCHES --custom-fa-branch b7d29fb"
# add for "gfx1201"
elif [[ "$ARCH" == "gfx1201" ]]; then
    echo "Building for RDNA3: gfx1201"
fi

export PYTORCH_ROCM_ARCH="gfx1201"
export TORCH_CUDA_ARCH_LIST="gfx1201"
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Create llama2-7b docker
bash $SETUP_DIR/build_docker.sh --base-docker-image $VLLM_IMAGE_NAME --arch $ARCH $PATCHES
image_name=$(cat mlperf_image_name.txt)
if [ ! -z "$1" ]; then docker tag $image_name $1; fi
