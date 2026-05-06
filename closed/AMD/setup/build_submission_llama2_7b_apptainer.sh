#!/bin/bash
set -e

SETUP_DIR=$(dirname -- "$0")

# ARCH=${ARCH:-$(rocminfo | grep "Name:" | grep "gfx" | awk 'NR==1' | awk '{print $2}')}
ARCH="gfx1201"

VLLM_IMAGE_NAME=rocm/vllm-dev:nightly_0610_rc2_0610_rc2_20250605
PATCHES="--apply-vllm-patches --apply-llama2-patches"

if [[ "$ARCH" == "gfx950" ]]; then
    VLLM_IMAGE_NAME=rocm/7.0-preview:rocm7.0_preview_ubuntu_22.04_vllm_0.8.5_mi35X_prealpha1
    PATCHES="$PATCHES --apply-aiter-patches"
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

# Create llama2-7b Apptainer image
bash "$SETUP_DIR/build_apptainer.sh" \
    --base-docker-image "$VLLM_IMAGE_NAME" \
    --arch "$ARCH" \
    $PATCHES

image_name=$(cat mlperf_apptainer_image_name.txt)

# Optional: if user passes an output sif path/name, copy the generated SIF to that path.
# Example:
#   bash setup/build_submission_apptainer_env.sh ./llama2_7b_gfx1201.sif
if [ -n "$1" ]; then
    output_name="$1"

    # If user gives a name without .sif, append .sif
    if [[ "$output_name" != *.sif ]]; then
        output_name="${output_name}.sif"
    fi

    cp "$image_name" "$output_name"
    image_name=$(readlink -f "$output_name")
fi

echo "$image_name" > mlperf_apptainer_image_name.txt
echo "$image_name" > mlperf_image_name.txt

echo "Apptainer image: $image_name"
echo "Run with:"
echo "apptainer exec --rocm \"$image_name\" bash"