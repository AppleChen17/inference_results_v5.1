#!/bin/bash
export OPENORCA_PARQUET="open_orca_dataset.parquet"
export DOWNLOAD_DIR="/data/openorca-dataset"
export EXPORT_DIR="/data/processed-openorca"
# export MODEL_PATH="/model/llama2-70b-chat-hf/orig"
export MODEL_PATH="/model/llama2-7b-chat-hf/orig"
export DATASET_PATH="${DOWNLOAD_DIR}/${OPENORCA_PARQUET}"

# Process the dataset according the Taskforce's agreed criteria
mkdir -p ${DOWNLOAD_DIR}
wget -O "${DATASET_PATH}" https://huggingface.co/datasets/Open-Orca/OpenOrca/resolve/main/1M-GPT4-Augmented.parquet?download=true
python3 processorca.py --dataset_pq_path=${DATASET_PATH} --model_dir=${MODEL_PATH} --seqlen_limit=1024 --export_dir=${EXPORT_DIR} --num_total_samples=24576
