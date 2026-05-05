#!/bin/bash

# set -x
set -e

PYTHON3_PATH=/usr/bin/python3
DATASET_FILE_PATH=/data/processed-openorca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl
MODEL_PATH=/model/llama2-70b-chat-hf/fp8_quantized
EVAL_SCRIPT_PATH=/app/mlperf_inference/language/llama2-70b/evaluate-accuracy.py

ARCH=${ARCH:-$(rocminfo | grep "Name:" | grep "gfx" | awk 'NR==1' | awk '{print $2}')}
if [[ "$ARCH" == "gfx950" ]]; then
    PYTHON3_PATH=/opt/conda/envs/py_3.10/bin/python
    MODEL_PATH=/model/llama2-70b-chat-hf/fp4_quantized
fi

if [ ! -f ${PYTHON3_PATH} ]; then
    echo "Wrong python3 path"
    exit 1
fi

if [ ! -f ${DATASET_FILE_PATH} ]; then
    echo "dataset not found, check the README.md how to download it"
    exit 1
fi

if [ ! -d ${MODEL_PATH} ]; then
    echo "model not found, check the README.md how to download it"
    exit 1
elif [ -z "$(ls -A ${MODEL_PATH})" ]; then
    echo "model dir is empty, check the README.md how to download it"
    exit 1
fi

if [ ! -f ${EVAL_SCRIPT_PATH} ]; then
    echo "tools/evaluate-accuracy.py not found"
    exit 1
fi

ACCURACY_JSON=${1}

if [ -z ${ACCURACY_JSON} ]; then
    echo "incorrect accuracy path, set it with ${0} <path>"
    exit 1
fi

if [ ! -f ${ACCURACY_JSON} ]; then
    echo "incorrect accuracy path, set it with ${0} <path>"
    exit 1
fi

OUTPUT_DIR=$(dirname ${ACCURACY_JSON})
RESULT_TXT=${OUTPUT_DIR}/accuracy.txt

# Pre-download datasets to avoid multithreading issues
${PYTHON3_PATH} -c 'import evaluate; evaluate.load("rouge"); import nltk; nltk.download("punkt"); nltk.download("punkt_tab")'

${PYTHON3_PATH} -u ${EVAL_SCRIPT_PATH} --checkpoint-path ${MODEL_PATH} \
                                       --mlperf-accuracy-file ${ACCURACY_JSON} \
                                       --dataset-file ${DATASET_FILE_PATH} \
                                       --dtype int32 > ${RESULT_TXT}

echo "Check $RESULT_TXT for the accuracy scores"
