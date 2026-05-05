#!/bin/bash
set -xeu

export LAB_TS=`date +%m%d-%H%M`
export LAB_MODEL="${LAB_MODEL:-/data/inference/model/}"
export LAB_DATASET="${LAB_DATASET:-/data/inference/data/}"

export LAB_DKR_CTNAME_BASE=mlperf.model_and_dataset.$(whoami)
export LAB_DKR_CTNAME=${LAB_DKR_CTNAME_BASE}.${LAB_TS}

EXTRA_ARGS=${EXTRA_ARGS:-''}

docker run ${EXTRA_ARGS} --rm -it --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
        --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        --name=${LAB_DKR_CTNAME} \
        -v ${LAB_MODEL}:/model/ \
        -v ${LAB_DATASET}:/data/ \
        mlperf_inference_submission_model_and_dataset_prep:5.1
