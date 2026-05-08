#!/bin/bash
set -xeu

IMAGE_NAME="${1:-}"

if [ -z "$IMAGE_NAME" ]; then
  echo "Error: Missing mandatory argument."
  echo "Usage: $0 <APPTAINER_IMAGE_NAME_OR_SIF_PATH>"
  exit 1
fi

# Support Docker-like image name:
#   test-mlperf-apptainer
# resolves to:
#   test-mlperf-apptainer.sif
if [ ! -f "$IMAGE_NAME" ] && [ -f "${IMAGE_NAME}.sif" ]; then
  IMAGE_NAME="${IMAGE_NAME}.sif"
fi

if [ ! -f "$IMAGE_NAME" ]; then
  echo "Error: Apptainer SIF image does not exist: $IMAGE_NAME"
  echo "If you used a short name, expected file: ${IMAGE_NAME}.sif"
  exit 1
fi

IMAGE_NAME=$(readlink -f "$IMAGE_NAME")

export LAB_TS=$(date +%m%d-%H%M%S)

# Host side
export LAB_MLPINF=$(dirname "$(dirname "$(readlink -fm -- "$0")")")
export LAB_MLPINF_CODE=${LAB_MLPINF}/code
export LAB_MLPINF_SUBMISSION=${LAB_MLPINF}/submission
export LAB_MLPINF_SETUP=${LAB_MLPINF}/setup

export LAB_MODEL="${LAB_MODEL:-/data/inference/model/}"
export LAB_DATASET="${LAB_DATASET:-/data/inference/data/}"

# Apptainer bind source paths must exist.
# Docker sometimes creates missing -v source directories automatically;
# Apptainer does not.
mkdir -p "${LAB_MLPINF_SUBMISSION}"

if [ ! -d "${LAB_MLPINF_CODE}" ]; then
  echo "Error: code directory does not exist: ${LAB_MLPINF_CODE}"
  exit 1
fi

if [ ! -d "${LAB_MLPINF_SETUP}" ]; then
  echo "Error: setup directory does not exist: ${LAB_MLPINF_SETUP}"
  exit 1
fi

if [ ! -d "${LAB_MODEL}" ]; then
  echo "Error: model directory does not exist: ${LAB_MODEL}"
  exit 1
fi

if [ ! -d "${LAB_DATASET}" ]; then
  echo "Error: dataset directory does not exist: ${LAB_DATASET}"
  exit 1
fi


# Normalize visible-device env for ROCm/vLLM.
# vLLM ROCm asserts HIP_VISIBLE_DEVICES == CUDA_VISIBLE_DEVICES if both exist.
if [ -z "${HIP_VISIBLE_DEVICES:-}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export HIP_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${HIP_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
fi

if [ -z "${ROCR_VISIBLE_DEVICES:-}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export ROCR_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi

if [ -z "${GPU_DEVICE_ORDINAL:-}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export GPU_DEVICE_ORDINAL="${CUDA_VISIBLE_DEVICES}"
fi

EXTRA_ARGS=${EXTRA_ARGS:-''}

ENV_ARGS=(
  --env LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64"
)


# apptainer exec \
#   --cleanenv \
#   ${EXTRA_ARGS} \
#   "${ENV_ARGS[@]}" \
#   -B /etc/passwd:/etc/passwd:ro \
#   -B /etc/group:/etc/group:ro \
#   -B "${LAB_MODEL}:/model/" \
#   -B "${LAB_DATASET}:/data/" \
#   -B "${LAB_MLPINF_CODE}:/lab-mlperf-inference/code" \
#   -B "${LAB_MLPINF_SUBMISSION}:/lab-mlperf-inference/submission" \
#   -B "${LAB_MLPINF_SETUP}:/lab-mlperf-inference/setup" \
#   "$IMAGE_NAME" \
#   bash

apptainer exec \
  --cleanenv \
  ${EXTRA_ARGS} \
  "${ENV_ARGS[@]}" \
  -B /etc/passwd:/etc/passwd:ro \
  -B /etc/group:/etc/group:ro \
  -B "${LAB_MODEL}:/model/" \
  -B "${LAB_DATASET}:/data/" \
  -B "${LAB_MLPINF_CODE}:/lab-mlperf-inference/code" \
  -B "${LAB_MLPINF_SUBMISSION}:/lab-mlperf-inference/submission" \
  -B "${LAB_MLPINF_SETUP}:/lab-mlperf-inference/setup" \
  -B "${LAB_MLPINF}/results:/lab-mlperf-inference/results" \
  "$IMAGE_NAME" \
  bash --noprofile --norc