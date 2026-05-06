#!/bin/bash
set -xeu

IMAGE_NAME="${1:-}"

if [ -z "$IMAGE_NAME" ]; then
  echo "Error: Missing mandatory argument."
  echo "Usage: $0 <APPTAINER_SIF_IMAGE>"
  exit 1
fi

if [ ! -f "$IMAGE_NAME" ]; then
  echo "Error: Apptainer SIF image does not exist: $IMAGE_NAME"
  exit 1
fi

export LAB_TS=$(date +%m%d-%H%M)

export LAB_MLPINF=$(dirname "$(dirname "$(readlink -fm -- "$0")")")
export LAB_MLPINF_CODE=${LAB_MLPINF}/code
export LAB_MLPINF_SUBMISSION=${LAB_MLPINF}/submission
export LAB_MLPINF_SETUP=${LAB_MLPINF}/setup

export LAB_MODEL="${LAB_MODEL:-/data/inference/model/}"
export LAB_DATASET="${LAB_DATASET:-/data/inference/data/}"

EXTRA_ARGS=${EXTRA_ARGS:-''}

apptainer exec \
  --rocm \
  --cleanenv \
  ${EXTRA_ARGS} \
  -B "${LAB_MODEL}:/model/" \
  -B "${LAB_DATASET}:/data/" \
  -B "${LAB_MLPINF_CODE}:/lab-mlperf-inference/code" \
  -B "${LAB_MLPINF_SUBMISSION}:/lab-mlperf-inference/submission" \
  -B "${LAB_MLPINF_SETUP}:/lab-mlperf-inference/setup" \
  "$IMAGE_NAME" \
  bash