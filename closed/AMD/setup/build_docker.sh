#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Terminate at first error
set -e

show_help() {
  echo "
  Usage: ${0##*/} [OPTIONS] [DOCKER_IMAGE_NAME]

  This script builds a custom Docker image, allowing for customizations
  of the vLLM and aiter libraries from specified git branches and patches.

  Options:
    -h, --help                          Display this help message and exit.

  Image Configuration:
    --base-docker-image <image>         Specify the base Docker image to build from.
                                        (e.g., 'rocm/vllm-dev:nightly_0715_rc1_0715_rc1_20250701')
    --image-name-postfix <postfix>      Append a postfix to the final Docker image name.
    --remove-git-folder                 Remove the .git folder from cloned libraries
    --arch <architecture>               Specify the target gpu architecture (e.g., gfx942, gfx950).

  vLLM Customization:
    --custom-vllm-branch <branch>       Specify a custom git branch or commit for vLLM.
    --apply-vllm-patches                Apply general custom patches to vLLM.
    --apply-llama2-patches              Apply custom Llama2-specific patches to vLLM.
    --apply-moe-patches                 Apply custom Mixture of Experts (MoE) patches to vLLM.
    --custom-vllm-patches-applied       (Internal flag) Indicates vLLM patches have been applied.

  aiter Customization:
    --custom-aiter-branch <branch>      Specify a custom git branch or commit for aiter.
    --apply-aiter-patches               Apply custom patches to aiter.
    --custom-aiter-patches-applied      (Internal flag) Indicates aiter patches have been applied.

  Flash Attention Customization:
      --custom-fa-branch <branch>      Specify a custom git branch or commit for Flash Attention.

  Positional Arguments:
    DOCKER_IMAGE_NAME                   Specify the base Docker image to build from
                                        (e.g., 'rocm/vllm-dev:nightly_0715_rc1_0715_rc1_20250701').
                                        If --base-docker-image is also set, that flag takes precedence.
  "
}

error_exit() {
  echo -e "${RED}Error: $1${NC}" >&2
  echo -e "${RED}Use '${0##*/} --help' for a list of available options.${NC}" >&2
  exit 1
}

get_commit_hash() {
  local DIR=$1
  pushd "$DIR" > /dev/null
  local COMMIT_HASH=$(git rev-parse --short HEAD)
  local GIT_EXIT_STATUS=$?
  popd > /dev/null
  if [ $GIT_EXIT_STATUS -ne 0 ] || [ -z "$COMMIT_HASH" ]; then
    COMMIT_HASH=$(date +"%Y%m%d")
  fi
  echo "$COMMIT_HASH"
}

remove_git_folder() {
  local DIR=$1
  if [ -n "$REMOVE_GIT_FOLDER" ]; then
    echo -e "${YELLOW}Remove .git folder from $DIR${NC}"
    rm -Rf "$DIR/.git"
  fi
}

# Cmdline argument handling
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
      shift 1
      ;;
    --arch)
      ARCH="$2"
      shift 2
      ;;
    -*|--*)
      error_exit "Unknown option $1"
      ;;
    *)
      DOCKER_IMAGE_NAME=$1
      shift
      ;;
  esac
done

SETUP_DIR=$(dirname -- $0)

# Use the specified image as the base
if [ -n "$DOCKER_IMAGE_NAME" ]; then
    BASE_IMAGE_NAME="$DOCKER_IMAGE_NAME"
fi

if [ -n "$DOCKER_IMAGE_FROM_FLAG" ]; then
    BASE_IMAGE_NAME="$DOCKER_IMAGE_FROM_FLAG"
fi

if [ -z "$BASE_IMAGE_NAME" ]; then
    error_exit "vLLM docker image is not specified"
fi


if [[ -z "$ARCH" ]]; then
  ARCH=$(rocminfo | grep "Name:" | grep "gfx" | awk 'NR==1' | awk '{print $2}')
fi

if [[ "$ARCH" != "gfx942" && "$ARCH" != "gfx950" && "$ARCH" != "gfx1201" ]]; then
  error_exit "Error: unsupported arch=$ARCH (supported: gfx942,gfx950)"
fi

# aiter-related preparations
BUILD_AITER=0
if [ -n "$CUSTOM_AITER_BRANCH" ] || [ -n "$CUSTOM_AITER_PATCHES" ] || [ -n "$CUSTOM_AITER_APPLIED" ]; then
    BUILD_AITER=1
fi

AITER_COMMIT_HASH=""
if [ "$BUILD_AITER" -eq "1" ] && [ -z "$CUSTOM_AITER_APPLIED" ]; then
    AITER_DIR="$SETUP_DIR/aiter"
    $SETUP_DIR/ensure_aiter_branch.sh "$BASE_IMAGE_NAME" "$AITER_DIR" "$ARCH" "$CUSTOM_AITER_BRANCH" "$CUSTOM_AITER_PATCHES"
    AITER_COMMIT_HASH=$(get_commit_hash $AITER_DIR)
    remove_git_folder $AITER_DIR
fi

# vllm-related preparations
VLLM_DIR="$SETUP_DIR/vllm"
$SETUP_DIR/ensure_vllm_branch.sh "$BASE_IMAGE_NAME" "$VLLM_DIR" "$ARCH" "$CUSTOM_VLLM_BRANCH" "$CUSTOM_VLLM_PATCHES" "$CUSTOM_LLAMA2_PATCHES" "$CUSTOM_MOE_PATCHES"
VLLM_COMMIT_HASH=$(get_commit_hash $VLLM_DIR)
remove_git_folder $VLLM_DIR

# Extract the tag part from the image name
RELEASE_TAG=${BASE_IMAGE_NAME##*:}
MLPERF_IMAGE_NAME="rocm/mlperf-inference:${RELEASE_TAG}-mlperf"

INSTALL_TORCHAO=0
if [ -n "$CUSTOM_LLAMA2_PATCHES" ] && [ "$ARCH" == "gfx942" ] || [ "$ARCH" == "gfx1201" ]; then
    INSTALL_TORCHAO=1
fi
# Create base mlperf docker image
docker build --build-arg BASE_IMAGE=${BASE_IMAGE_NAME} --build-arg INSTALL_TORCHAO=${INSTALL_TORCHAO} -f "$SETUP_DIR/Dockerfile.mlperf_$ARCH" -t ${MLPERF_IMAGE_NAME} "$SETUP_DIR/.."

# Add aiter to the docker image
if [ "$BUILD_AITER" -eq "1" ]; then
    IMAGE_NAME="$MLPERF_IMAGE_NAME-custom-aiter"
    # TODO This avoids aiter re-build, avoid this later
    if [ "$ARCH" == "gfx950" ]; then
      AITER_DOCKERFILE="$SETUP_DIR/Dockerfile.override_aiter"
    else
      AITER_DOCKERFILE="$SETUP_DIR/Dockerfile.aiter"
    fi
    docker build --build-arg BASE_IMAGE=${MLPERF_IMAGE_NAME} -f ${AITER_DOCKERFILE} -t ${IMAGE_NAME} "$SETUP_DIR/.."
    MLPERF_IMAGE_NAME="$IMAGE_NAME"
fi

# Install custom flash attention to the docker image, e.g. for 2.7.2 pass --custom-fa-branch b7d29fb
if [ -n "$CUSTOM_FA_BRANCH" ]; then
    IMAGE_NAME="$MLPERF_IMAGE_NAME-custom-fa"
    docker build --build-arg BASE_IMAGE=${MLPERF_IMAGE_NAME} --build-arg ARCH=${ARCH} --build-arg FA_BRANCH=${CUSTOM_FA_BRANCH} -f "$SETUP_DIR/Dockerfile.fa" -t ${IMAGE_NAME} "$SETUP_DIR/.."
    MLPERF_IMAGE_NAME="$IMAGE_NAME"
fi

# Add vllm to the docker image
IMAGE_NAME="$MLPERF_IMAGE_NAME-custom-vllm"
docker build --build-arg BASE_IMAGE=${MLPERF_IMAGE_NAME} --build-arg ARCH=${ARCH} -f "$SETUP_DIR/Dockerfile.vllm" -t ${IMAGE_NAME} "$SETUP_DIR/.."
MLPERF_IMAGE_NAME="$IMAGE_NAME"

# Construct the final Docker image name
HARNESS_COMMIT_HASH=$(get_commit_hash ".")
FINAL_IMAGE_NAME="rocm/mlperf-inference:${RELEASE_TAG}__h-${HARNESS_COMMIT_HASH}_v-${VLLM_COMMIT_HASH}"

rm -rf $VLLM_DIR
if [ -n "$AITER_COMMIT_HASH" ]; then
  rm -rf $AITER_DIR
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

docker tag $MLPERF_IMAGE_NAME $FINAL_IMAGE_NAME

echo "${FINAL_IMAGE_NAME}" > mlperf_image_name.txt
echo -e "${GREEN}Run the following command to start a container: bash setup/start.sh ${FINAL_IMAGE_NAME}${NC}"
