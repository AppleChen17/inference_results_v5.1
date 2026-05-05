#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

function get_vllm_version_hash() {
    # Extract vllm version hash from the package version like '0.6.7.dev1+g3efdd2be.rocm630'
    DOCKER_IMAGE_NAME=$1
    LINE_COMMIT_ID=$(docker run --rm $DOCKER_IMAGE_NAME pip list | grep '^\bvllm\b' | awk '{print $2}')
    # Regex to extract the vllm version hash
    REGEX_COMMIT_ID=".*\+g(\w+).*"

    if [[ $LINE_COMMIT_ID =~ $REGEX_COMMIT_ID ]]
    then
        COMMIT_ID="${BASH_REMATCH[1]}"
        # Return the commit hash
        echo "$COMMIT_ID"
    fi
}

function apply_patches() {
    local patch_list_file="$1"
    local patch_folder="$2"

    while IFS= read -r patch_file || [ -n "$patch_file" ]; do
        patch_file=$(echo "$patch_file" | sed 's/[[:space:]]*$//')

        if [[ -z "$patch_file" || "$patch_file" == \#* ]]; then
            echo -e "${YELLOW}Patch file skipped: $patch_file ${NC}"
            continue
        fi

        if [[ ! -f "$patch_folder/$patch_file" ]]; then
            echo -e "${RED}Patch file '$patch_folder/$patch_file' not found. ${NC}"
            exit 1
        fi

        echo -e "${GREEN}Applying patch '$patch_folder/$patch_file'... ${NC}"
        git apply "$patch_folder/$patch_file"

        if [[ $? -ne 0 ]]; then
            echo -e "${RED}Failed to apply patch '$patch_folder/$patch_file'. ${NC}"
            exit 1
        fi

    done < "$patch_list_file"
}

DOCKER_IMAGE_NAME=$1
VLLM_DIR=$2
ARCH=$3
CUSTOM_BRANCH=$4
CUSTOM_VLLM_PATCHES=$5
CUSTOM_LLAMA2_PATCHES=$6
CUSTOM_MOE_PATCHES=$7

echo -e "${GREEN}Set VLLM repository ${NC}"

# Pull the specified image and retrieve the vllm version
docker pull "$DOCKER_IMAGE_NAME"
VLLM_COMMIT=$(get_vllm_version_hash "$DOCKER_IMAGE_NAME")

if [ -n "$CUSTOM_BRANCH" ]; then
    VLLM_COMMIT="$CUSTOM_BRANCH"
fi

echo -e "${GREEN}VLLM git commit: ${VLLM_COMMIT} ${NC}"

if [ -d "$VLLM_DIR" ]; then
    echo -e "${YELLOW}Remove existing vllm dir: ${VLLM_DIR} ${NC}"
    rm -rf $VLLM_DIR
fi

# TODO remove this when the gfx950 commit is available on rocm/vllm
if [ "$ARCH" == "gfx950" ]; then
    VLLM_CONTAINER_NAME="get-vllm-folder"
    docker run --name $VLLM_CONTAINER_NAME -d $DOCKER_IMAGE_NAME
    docker cp $VLLM_CONTAINER_NAME:/app/vllm "$VLLM_DIR"
    docker stop $VLLM_CONTAINER_NAME
    docker rm $VLLM_CONTAINER_NAME
    rm -fr "$VLLM_DIR"/build
else
    git clone --filter=blob:none https://github.com/ROCm/vllm.git "$VLLM_DIR"
fi

# Make it absolute to work properly with subsequent script calls
VLLM_DIR=$(readlink -e $VLLM_DIR)

# Switch to the commit
git -C "$VLLM_DIR" checkout $VLLM_COMMIT

cd "$SCRIPT_DIR/vllm/"

PATCH_FILE_FOLDER="$SCRIPT_DIR/vllm_patches"

if [ -n "$CUSTOM_VLLM_PATCHES" ]; then
    echo -e "${GREEN}Apply common patches.. ${NC}"
    COMMON_PATCH_FILES="$PATCH_FILE_FOLDER/common_patch_files.txt"
    apply_patches $COMMON_PATCH_FILES $PATCH_FILE_FOLDER
fi

if [ -n "$CUSTOM_LLAMA2_PATCHES" ]; then
    echo -e "${GREEN}Apply llama2 patches.. ${NC}"
    LLAMA2_PATCH_FILES="$PATCH_FILE_FOLDER/llama2_patch_files_${ARCH}.txt"
    apply_patches $LLAMA2_PATCH_FILES $PATCH_FILE_FOLDER
fi

if [ -n "$CUSTOM_MOE_PATCHES" ]; then
    echo -e "${GREEN}Apply moe patches.. ${NC}"
    MOE_PATCH_FILES="$PATCH_FILE_FOLDER/moe_patch_files.txt"
    apply_patches $MOE_PATCH_FILES $PATCH_FILE_FOLDER
fi

echo -e "${GREEN}All patches applied successfully ${NC}"
