#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

set -e

show_help() {
  echo "
  Usage: ${0##*/} [OPTIONS] [BASE_IMAGE_NAME]

  This script builds a custom Apptainer SIF image for MLPerf on Ubuntu 24.04.
  ... (其餘幫助內容省略) ...
  "
}

error_exit() {
  echo -e "${RED}Error: $1${NC}" >&2
  exit 1
}

get_commit_hash() {
  local DIR="$1"
  pushd "$DIR" > /dev/null
  local COMMIT_HASH
  COMMIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || true)
  popd > /dev/null
  if [ -z "$COMMIT_HASH" ]; then COMMIT_HASH=$(date +"%Y%m%d"); fi
  echo "$COMMIT_HASH"
}

remove_git_folder() {
  local DIR="$1"
  if [ -n "${REMOVE_GIT_FOLDER:-}" ]; then
    rm -rf "$DIR/.git"
  fi
}

sanitize_name() {
  echo "$1" | sed 's#[/:]#_#g' | sed 's#[^A-Za-z0-9_.-]#_#g'
}

# -----------------------------
# Parse command line arguments
# -----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-docker-image) DOCKER_IMAGE_FROM_FLAG="$2"; shift 2 ;;
    --custom-vllm-branch) CUSTOM_VLLM_BRANCH="$2"; shift 2 ;;
    --apply-vllm-patches) CUSTOM_VLLM_PATCHES=1; shift ;;
    --apply-llama2-patches) CUSTOM_LLAMA2_PATCHES=1; shift ;;
    --apply-moe-patches) CUSTOM_MOE_PATCHES=1; shift ;;
    --arch) ARCH="$2"; shift 2 ;;
    *) DOCKER_IMAGE_NAME="$1"; shift ;;
  esac
done

SETUP_DIR=$(dirname -- "$0")
SETUP_DIR=$(readlink -f "$SETUP_DIR")
REPO_ROOT=$(readlink -f "$SETUP_DIR/..")

# Resolve Base Image
BASE_IMAGE_NAME="${DOCKER_IMAGE_FROM_FLAG:-$DOCKER_IMAGE_NAME}"
if [ -z "$BASE_IMAGE_NAME" ]; then error_exit "Base Docker image not specified"; fi

# Resolve Architecture
if [[ -z "${ARCH:-}" ]]; then
  ARCH=$(rocminfo | grep "Name:" | grep "gfx" | awk 'NR==1' | awk '{print $2}')
fi

echo -e "${GREEN}Target arch: $ARCH${NC}"
echo -e "${GREEN}Base image:  $BASE_IMAGE_NAME${NC}"

export PYTORCH_ROCM_ARCH="$ARCH"
export TORCH_CUDA_ARCH_LIST="$ARCH"
if [[ "$ARCH" == "gfx1201" ]]; then export HSA_OVERRIDE_GFX_VERSION="11.0.0"; fi

# -----------------------------
# Build Steps Start
# -----------------------------
VLLM_DIR="$SETUP_DIR/vllm"
"$SETUP_DIR/ensure_vllm_branch.sh" "$BASE_IMAGE_NAME" "$VLLM_DIR" "$ARCH" "${CUSTOM_VLLM_BRANCH:-}" "${CUSTOM_VLLM_PATCHES:-}" "${CUSTOM_LLAMA2_PATCHES:-}"
VLLM_COMMIT_HASH=$(get_commit_hash "$VLLM_DIR")
remove_git_folder "$VLLM_DIR"

RELEASE_TAG=${BASE_IMAGE_NAME##*:}
FINAL_IMAGE_NAME=$(sanitize_name "rocm_mlperf_${RELEASE_TAG}_${ARCH}")
BUILD_DIR="$REPO_ROOT/.apptainer_build"
SANDBOX="$BUILD_DIR/${FINAL_IMAGE_NAME}.sandbox"
SIF_PATH="$REPO_ROOT/${FINAL_IMAGE_NAME}.sif"

mkdir -p "$BUILD_DIR"
rm -rf "$SANDBOX"

# 1. Build Sandbox
echo -e "${GREEN}=== Build Apptainer sandbox ===${NC}"
apptainer build --fakeroot --sandbox "$SANDBOX" "docker://$BASE_IMAGE_NAME"

# 定義在 Sandbox 中執行的函式
run_in_sandbox() {
  apptainer exec --fakeroot --writable \
    --env PYTORCH_ROCM_ARCH="$PYTORCH_ROCM_ARCH" \
    --env HSA_OVERRIDE_GFX_VERSION="$HSA_OVERRIDE_GFX_VERSION" \
    "$SANDBOX" bash -lc "$1"
}

# 2. Install Apt Dependencies
echo -e "${GREEN}=== Install MLPerf base apt dependencies ===${NC}"
run_in_sandbox "
apt update
apt install -y libfmt-dev libsqlite3-dev numactl libnuma-dev sqlite3 zip nano libcurl4-openssl-dev build-essential git
rm -rf /var/lib/apt/lists/*
"

# 3. Install Python Dependencies
echo -e "${GREEN}=== Install Python dependencies ===${NC}"
run_in_sandbox "
python3 -m pip install absl-py==2.1.0 datasets==2.20.0 nltk==3.8.1 numpy==1.26.4 py-libnuma==1.2 rouge_score==0.1.2 omegaconf==2.3.0 optuna==4.1.0 evaluate==0.4.3
"

# 4. Install LoadGen & ROCm Tools
echo -e "${GREEN}=== Install MLPerf LoadGen & ROCm Tools ===${NC}"
run_in_sandbox "
git clone https://github.com/mlcommons/inference.git /app/mlperf_inference
cd /app/mlperf_inference/loadgen && git checkout 091d51f && python3 -m pip install .
"

# 5. Copy Code and Install Custom Packages
IMAGE_SRC="/opt/mlperf-src"
mkdir -p "$SANDBOX$IMAGE_SRC"
cp -a "$REPO_ROOT/code" "$SANDBOX/lab-mlperf-inference/code"
# 將整個 setup 目錄複製進去供編譯使用
cp -a "$REPO_ROOT/setup" "$SANDBOX$IMAGE_SRC/setup"

# 6. Install torchao (針對 gfx1201/24.04 優化)
INSTALL_TORCHAO=0
if [[ "$ARCH" == "gfx1201" ]]; then INSTALL_TORCHAO=1; fi

if [ "$INSTALL_TORCHAO" -eq 1 ]; then
  echo -e "${GREEN}=== Install torchao from source ===${NC}"
  run_in_sandbox "
    git clone https://github.com/pytorch/ao.git /app/ao
    cd /app/ao && git checkout 2e2ce0b8 && USE_CPP=1 python3 -m pip install -e . --no-build-isolation
  "
fi

# 7. Install custom vLLM
echo -e "${GREEN}=== Install custom vLLM ===${NC}"
run_in_sandbox "
  cd $IMAGE_SRC/setup/vllm
  python3 -m pip uninstall -y vllm || true
  export PYTORCH_ROCM_ARCH=$ARCH
  python3 -m pip install -v -e . --no-build-isolation
"

# 8. Sanity Check
echo -e "${GREEN}=== Sanity Check ===${NC}"
run_in_sandbox "python3 -c 'import torch; import vllm; print(\"Torch:\", torch.__version__); print(\"vLLM:\", vllm.__version__)'"

# 9. Convert to SIF
echo -e "${GREEN}=== Build final SIF ===${NC}"
rm -f "$SIF_PATH"
apptainer build --fakeroot "$SIF_PATH" "$SANDBOX"

echo -e "${GREEN}Done! Image at: $SIF_PATH${NC}"