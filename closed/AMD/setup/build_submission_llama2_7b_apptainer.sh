#!/bin/bash
set -e

SETUP_DIR=$(dirname -- "$0")

ARCH="gfx1201"

VLLM_IMAGE_NAME=rocm/vllm-dev:nightly_0610_rc2_0610_rc2_20250605

# From successful build log:
# vLLM git commit: 71faa1880
VLLM_BRANCH="${VLLM_BRANCH:-71faa1880}"

PATCHES="--custom-vllm-branch ${VLLM_BRANCH} --apply-vllm-patches --apply-llama2-patches"

if [[ "$ARCH" == "gfx950" ]]; then
    VLLM_IMAGE_NAME=rocm/7.0-preview:rocm7.0_preview_ubuntu_22.04_vllm_0.8.5_mi35X_prealpha1

    # gfx950 may need aiter. Do not guess it here.
    # Set AITER_BRANCH manually if you need this path.
    if [[ -n "${AITER_BRANCH:-}" ]]; then
        PATCHES="$PATCHES --custom-aiter-branch ${AITER_BRANCH} --apply-aiter-patches"
    else
        PATCHES="$PATCHES --apply-aiter-patches"
    fi

elif [[ "$ARCH" == "gfx942" ]]; then
    PATCHES="$PATCHES --custom-fa-branch b7d29fb"

elif [[ "$ARCH" == "gfx1201" ]]; then
    echo "Building for RDNA3: gfx1201"
fi

export PYTORCH_ROCM_ARCH="$ARCH"
export TORCH_CUDA_ARCH_LIST="$ARCH"

if [[ "$ARCH" == "gfx1201" ]]; then
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
fi

bash "$SETUP_DIR/build_apptainer.sh" \
    --base-docker-image "$VLLM_IMAGE_NAME" \
    --arch "$ARCH" \
    $PATCHES

image_name=$(cat mlperf_apptainer_image_name.txt)

if [ -n "${1:-}" ]; then
    output_name="$1"

    if [[ "$output_name" != *.sif ]]; then
        output_name="${output_name}.sif"
    fi

    cp "$image_name" "$output_name"
    image_name="$output_name"
fi

echo "$image_name" > mlperf_apptainer_image_name.txt
echo "$image_name" > mlperf_image_name.txt

echo "Apptainer image: $image_name"
echo "Run with:"
echo "bash setup/start_submission_apptainer_env.sh $image_name"