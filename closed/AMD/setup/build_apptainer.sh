#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

set -e

show_help() {
  echo "
  Usage: ${0##*/} [OPTIONS] [BASE_IMAGE_NAME]

  This script builds a custom Apptainer SIF image without using Docker daemon.

  Options:
    -h, --help                          Display this help message and exit.

  Image Configuration:
    --base-docker-image <image>         Specify the base Docker image to build from.
                                        Example:
                                        rocm/vllm-dev:nightly_0610_rc2_0610_rc2_20250605

    --image-name-postfix <postfix>      Append a postfix to the final Apptainer image name.
    --remove-git-folder                 Remove the .git folder from cloned libraries.
    --arch <architecture>               Specify target GPU architecture.
                                        Supported: gfx942, gfx950, gfx1201

  vLLM Customization:
    --custom-vllm-branch <branch>       Specify a custom git branch or commit for vLLM.
    --apply-vllm-patches                Apply general custom patches to vLLM.
    --apply-llama2-patches              Apply custom Llama2-specific patches to vLLM.
    --apply-moe-patches                 Apply custom MoE patches to vLLM.
    --custom-vllm-patches-applied       Internal flag.

  aiter Customization:
    --custom-aiter-branch <branch>      Specify a custom git branch or commit for aiter.
    --apply-aiter-patches               Apply custom patches to aiter.
    --custom-aiter-patches-applied      Internal flag.

  Flash Attention Customization:
    --custom-fa-branch <branch>         Specify custom Flash Attention branch or commit.

  Environment variables:
    APPTAINER_BUILD_OPTS                Extra options for apptainer build.
                                        Default: --fakeroot

    APPTAINER_EXEC_OPTS                 Extra options for apptainer exec.
                                        Default: empty

    KEEP_APPTAINER_SANDBOX=1            Keep the writable sandbox after building SIF.

  Positional Arguments:
    BASE_IMAGE_NAME                     Base Docker image to pull via docker://.
  "
}

error_exit() {
  echo -e "${RED}Error: $1${NC}" >&2
  echo -e "${RED}Use '${0##*/} --help' for options.${NC}" >&2
  exit 1
}

get_commit_hash() {
  local DIR=$1
  pushd "$DIR" > /dev/null
  local COMMIT_HASH
  COMMIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || true)
  popd > /dev/null

  if [ -z "$COMMIT_HASH" ]; then
    COMMIT_HASH=$(date +"%Y%m%d")
  fi

  echo "$COMMIT_HASH"
}

remove_git_folder() {
  local DIR=$1
  if [ -n "$REMOVE_GIT_FOLDER" ]; then
    echo -e "${YELLOW}Remove .git folder from $DIR${NC}"
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
  case $1 in
    -h|--help)
      show_help
      exit 0
      ;;
    --base-docker-image)
      DOCKER_IMAGE_FROM_FLAG="$2"
      shift 2
      ;;
    --custom-vllm-branch)
      CUSTOM_VLLM_BRANCH="$2"
      shift 2
      ;;
    --apply-vllm-patches)
      CUSTOM_VLLM_PATCHES=1
      shift
      ;;
    --apply-llama2-patches)
      CUSTOM_LLAMA2_PATCHES=1
      shift
      ;;
    --apply-moe-patches)
      CUSTOM_MOE_PATCHES=1
      shift
      ;;
    --custom-vllm-patches-applied)
      CUSTOM_VLLM_APPLIED=1
      shift
      ;;
    --custom-aiter-branch)
      CUSTOM_AITER_BRANCH="$2"
      shift 2
      ;;
    --apply-aiter-patches)
      CUSTOM_AITER_PATCHES=1
      shift
      ;;
    --custom-aiter-patches-applied)
      CUSTOM_AITER_APPLIED=1
      shift
      ;;
    --custom-fa-branch)
      CUSTOM_FA_BRANCH="$2"
      shift 2
      ;;
    --image-name-postfix)
      IMAGE_NAME_POSTFIX="$2"
      shift 2
      ;;
    --remove-git-folder)
      REMOVE_GIT_FOLDER=1
      shift
      ;;
    --arch)
      ARCH="$2"
      shift 2
      ;;
    -*|--*)
      error_exit "Unknown option $1"
      ;;
    *)
      DOCKER_IMAGE_NAME="$1"
      shift
      ;;
  esac
done

SETUP_DIR=$(dirname -- "$0")
SETUP_DIR=$(readlink -f "$SETUP_DIR")
REPO_ROOT=$(readlink -f "$SETUP_DIR/..")

# -----------------------------
# Resolve base image
# -----------------------------

if [ -n "$DOCKER_IMAGE_NAME" ]; then
  BASE_IMAGE_NAME="$DOCKER_IMAGE_NAME"
fi

if [ -n "$DOCKER_IMAGE_FROM_FLAG" ]; then
  BASE_IMAGE_NAME="$DOCKER_IMAGE_FROM_FLAG"
fi

if [ -z "$BASE_IMAGE_NAME" ]; then
  error_exit "Base Docker image is not specified"
fi

# -----------------------------
# Resolve architecture
# -----------------------------

if [[ -z "$ARCH" ]]; then
  ARCH=$(rocminfo | grep "Name:" | grep "gfx" | awk 'NR==1' | awk '{print $2}')
fi

if [[ "$ARCH" != "gfx942" && "$ARCH" != "gfx950" && "$ARCH" != "gfx1201" ]]; then
  error_exit "Unsupported arch=$ARCH. Supported: gfx942, gfx950, gfx1201"
fi

echo -e "${GREEN}Target arch: $ARCH${NC}"
echo -e "${GREEN}Base image:  $BASE_IMAGE_NAME${NC}"

export PYTORCH_ROCM_ARCH="$ARCH"
export TORCH_CUDA_ARCH_LIST="$ARCH"

if [[ "$ARCH" == "gfx1201" ]]; then
  export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
fi

# -----------------------------
# Prepare aiter
# -----------------------------

BUILD_AITER=0

if [ -n "$CUSTOM_AITER_BRANCH" ] || [ -n "$CUSTOM_AITER_PATCHES" ] || [ -n "$CUSTOM_AITER_APPLIED" ]; then
  BUILD_AITER=1
fi

AITER_COMMIT_HASH=""

if [ "$BUILD_AITER" -eq 1 ] && [ -z "$CUSTOM_AITER_APPLIED" ]; then
  AITER_DIR="$SETUP_DIR/aiter"

  "$SETUP_DIR/ensure_aiter_branch.sh" \
    "$BASE_IMAGE_NAME" \
    "$AITER_DIR" \
    "$ARCH" \
    "$CUSTOM_AITER_BRANCH" \
    "$CUSTOM_AITER_PATCHES"

  AITER_COMMIT_HASH=$(get_commit_hash "$AITER_DIR")
  remove_git_folder "$AITER_DIR"
fi

# -----------------------------
# Prepare vLLM
# -----------------------------

VLLM_DIR="$SETUP_DIR/vllm"

"$SETUP_DIR/ensure_vllm_branch.sh" \
  "$BASE_IMAGE_NAME" \
  "$VLLM_DIR" \
  "$ARCH" \
  "$CUSTOM_VLLM_BRANCH" \
  "$CUSTOM_VLLM_PATCHES" \
  "$CUSTOM_LLAMA2_PATCHES" \
  "$CUSTOM_MOE_PATCHES"

VLLM_COMMIT_HASH=$(get_commit_hash "$VLLM_DIR")
remove_git_folder "$VLLM_DIR"

# -----------------------------
# Construct final image name
# -----------------------------

RELEASE_TAG=${BASE_IMAGE_NAME##*:}
HARNESS_COMMIT_HASH=$(get_commit_hash "$REPO_ROOT")

FINAL_IMAGE_NAME="rocm_mlperf-inference_${RELEASE_TAG}__h-${HARNESS_COMMIT_HASH}_v-${VLLM_COMMIT_HASH}"

if [ -n "$AITER_COMMIT_HASH" ]; then
  FINAL_IMAGE_NAME="${FINAL_IMAGE_NAME}_a-${AITER_COMMIT_HASH}"
fi

if [ -n "$CUSTOM_VLLM_PATCHES" ] || [ -n "$CUSTOM_LLAMA2_PATCHES" ] || [ -n "$CUSTOM_MOE_PATCHES" ]; then
  FINAL_IMAGE_NAME="${FINAL_IMAGE_NAME}_v-"

  if [ -n "$CUSTOM_VLLM_PATCHES" ]; then
    FINAL_IMAGE_NAME="${FINAL_IMAGE_NAME}c"
  fi

  if [ -n "$CUSTOM_LLAMA2_PATCHES" ]; then
    FINAL_IMAGE_NAME="${FINAL_IMAGE_NAME}l"
  fi

  if [ -n "$CUSTOM_MOE_PATCHES" ]; then
    FINAL_IMAGE_NAME="${FINAL_IMAGE_NAME}m"
  fi
fi

if [ -n "$CUSTOM_AITER_PATCHES" ]; then
  FINAL_IMAGE_NAME="${FINAL_IMAGE_NAME}_a-c"
fi

if [ -n "$CUSTOM_FA_BRANCH" ]; then
  FINAL_IMAGE_NAME="${FINAL_IMAGE_NAME}_fa-${CUSTOM_FA_BRANCH}"
fi

FINAL_IMAGE_NAME="${FINAL_IMAGE_NAME}_${ARCH}"

if [ -n "$IMAGE_NAME_POSTFIX" ]; then
  FINAL_IMAGE_NAME="${FINAL_IMAGE_NAME}_${IMAGE_NAME_POSTFIX}"
fi

FINAL_IMAGE_NAME=$(sanitize_name "$FINAL_IMAGE_NAME")

BUILD_DIR="$REPO_ROOT/.apptainer_build"
SANDBOX="$BUILD_DIR/${FINAL_IMAGE_NAME}.sandbox"
SIF_NAME="${FINAL_IMAGE_NAME}.sif"
SIF_PATH="$REPO_ROOT/$SIF_NAME"

mkdir -p "$BUILD_DIR"

# -----------------------------
# Apptainer helper
# -----------------------------

APPTAINER_BUILD_OPTS="${APPTAINER_BUILD_OPTS:---fakeroot}"
APPTAINER_EXEC_OPTS="${APPTAINER_EXEC_OPTS:-}"

run_in_sandbox() {
  local CMD="$1"

  apptainer exec \
    --writable \
    --rocm \
    $APPTAINER_EXEC_OPTS \
    --env PYTORCH_ROCM_ARCH="$PYTORCH_ROCM_ARCH" \
    --env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    --env HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-}" \
    "$SANDBOX" \
    bash -lc "$CMD"
}

# -----------------------------
# Build Apptainer sandbox
# -----------------------------

echo -e "${GREEN}=== Build Apptainer sandbox from docker:// image ===${NC}"

rm -rf "$SANDBOX"

apptainer build \
  $APPTAINER_BUILD_OPTS \
  --sandbox "$SANDBOX" \
  "docker://$BASE_IMAGE_NAME"

# -----------------------------
# Copy source tree into sandbox
# -----------------------------

echo -e "${GREEN}=== Copy MLPerf source tree into sandbox ===${NC}"

IMAGE_SRC="/opt/mlperf-src"
mkdir -p "$SANDBOX$IMAGE_SRC"

(
  cd "$REPO_ROOT"
  tar \
    --exclude=".git" \
    --exclude=".apptainer_build" \
    --exclude="*.sif" \
    -cf - .
) | (
  cd "$SANDBOX$IMAGE_SRC"
  tar -xf -
)

# -----------------------------
# Install MLPerf / vLLM / aiter inside sandbox
# -----------------------------

INSTALL_TORCHAO=0

# 修正原本 shell 條件的優先順序，避免 gfx1201 時邏輯不清楚
if { [ -n "$CUSTOM_LLAMA2_PATCHES" ] && [ "$ARCH" == "gfx942" ]; } || [ "$ARCH" == "gfx1201" ]; then
  INSTALL_TORCHAO=1
fi

echo -e "${GREEN}=== Install basic Python build tools ===${NC}"

run_in_sandbox "
set -e

python3 -m pip install --upgrade pip setuptools wheel packaging ninja cmake
"

if [ "$INSTALL_TORCHAO" -eq 1 ]; then
  echo -e "${GREEN}=== Install torchao ===${NC}"

  run_in_sandbox "
set -e

python3 -m pip install --no-cache-dir torchao
"
fi

if [ "$BUILD_AITER" -eq 1 ]; then
  echo -e "${GREEN}=== Install custom aiter ===${NC}"

  run_in_sandbox "
set -e

cd $IMAGE_SRC/setup/aiter

python3 -m pip uninstall -y aiter || true
python3 -m pip install -v . --no-build-isolation
"
fi

if [ -n "$CUSTOM_FA_BRANCH" ]; then
  echo -e "${GREEN}=== Install custom Flash Attention branch: $CUSTOM_FA_BRANCH ===${NC}"

  run_in_sandbox "
set -e

python3 -m pip uninstall -y flash-attn flash_attn || true

python3 -m pip install \
  --no-cache-dir \
  --no-build-isolation \
  \"git+https://github.com/ROCm/flash-attention.git@$CUSTOM_FA_BRANCH\" || \
python3 -m pip install \
  --no-cache-dir \
  --no-build-isolation \
  \"git+https://github.com/Dao-AILab/flash-attention.git@$CUSTOM_FA_BRANCH\"
"
fi

echo -e "${GREEN}=== Install custom vLLM ===${NC}"

run_in_sandbox "
set -e

cd $IMAGE_SRC/setup/vllm

python3 -m pip uninstall -y vllm || true
python3 -m pip install -v -e . --no-build-isolation
"

echo -e "${GREEN}=== Optional MLPerf requirements install ===${NC}"

run_in_sandbox "
set -e

cd $IMAGE_SRC

if [ -f requirements.txt ]; then
  python3 -m pip install -r requirements.txt
fi

if [ -f setup/requirements.txt ]; then
  python3 -m pip install -r setup/requirements.txt
fi
"

# -----------------------------
# Basic sanity check
# -----------------------------

echo -e "${GREEN}=== Sanity check inside Apptainer sandbox ===${NC}"

run_in_sandbox "
set -e

python3 - <<'PY'
import sys
print('Python:', sys.version)

try:
    import torch
    print('torch:', torch.__version__)
    print('torch.version.hip:', getattr(torch.version, 'hip', None))
except Exception as e:
    print('torch import failed:', repr(e))

try:
    import vllm
    print('vllm:', getattr(vllm, '__version__', 'unknown'))
except Exception as e:
    print('vllm import failed:', repr(e))
PY
"

# -----------------------------
# Clean temporary host-side cloned dirs
# -----------------------------

rm -rf "$VLLM_DIR"

if [ -n "$AITER_COMMIT_HASH" ]; then
  rm -rf "$AITER_DIR"
fi

# -----------------------------
# Convert sandbox to SIF
# -----------------------------

echo -e "${GREEN}=== Build final SIF ===${NC}"

rm -f "$SIF_PATH"

apptainer build \
  $APPTAINER_BUILD_OPTS \
  "$SIF_PATH" \
  "$SANDBOX"

if [ -z "$KEEP_APPTAINER_SANDBOX" ]; then
  rm -rf "$SANDBOX"
else
  echo -e "${YELLOW}Keep sandbox: $SANDBOX${NC}"
fi

echo "$SIF_PATH" > "$REPO_ROOT/mlperf_apptainer_image_name.txt"

# 為了讓舊腳本如果還讀 mlperf_image_name.txt，也能讀到 SIF path
echo "$SIF_PATH" > "$REPO_ROOT/mlperf_image_name.txt"

echo -e "${GREEN}Done.${NC}"
echo -e "${GREEN}Apptainer image:${NC} $SIF_PATH"
echo -e "${GREEN}Run:${NC}"
echo "apptainer exec --rocm $SIF_PATH bash"