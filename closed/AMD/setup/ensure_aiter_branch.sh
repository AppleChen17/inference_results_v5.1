#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

function get_aiter_version_hash() {
    # Extract aiter version hash from the package version like '0.1.3.dev40+g626d8127'
    DOCKER_IMAGE_NAME=$1
    LINE_COMMIT_ID=$(docker run --rm $DOCKER_IMAGE_NAME pip list | grep '^\baiter\b' | awk '{print $2}')
    # Regex to extract the aiter version hash
    REGEX_COMMIT_ID=".*\+g(\w+).*"

    if [[ $LINE_COMMIT_ID =~ $REGEX_COMMIT_ID ]]
    then
        COMMIT_ID="${BASH_REMATCH[1]}"
        # Return the commit hash
        echo "$COMMIT_ID"
    fi
}

declare -A module_path
module_path=(
    [aiter]="."
    [ck]="3rdparty/composable_kernel"
)

function apply_patches() {
    local patch_list_file="$1"  
    local patch_folder="$2"  

    while IFS= read -r line || [ -n "$line" ]; do
        line=$(echo "$line" | sed 's/[[:space:]]*$//')

        if [[ -z "$line" || "$line" == \#* ]]; then
            echo -e "${YELLOW}Patch file skipped: $line ${NC}"
            continue  
        fi

        module="${line%% *}"
        patch_file="${line#* }"

        pushd "${module_path[$module]}" > /dev/null

        if [[ ! -f "$patch_folder/$patch_file" ]]; then  
            echo -e "${RED}Patch file '$patch_folder/$patch_file' not found. ${NC}"  
            exit 1  
        fi

        echo -e "${GREEN}Applying patch '$module' '$patch_folder/$patch_file'... ${NC}"  
        git apply "$patch_folder/$patch_file"

        if [[ $? -ne 0 ]]; then  
            echo -e "${RED}Failed to apply patch '$patch_folder/$patch_file'. ${NC}"
            exit 1  
        fi 

        popd > /dev/null

    done < "$patch_list_file"
}

DOCKER_IMAGE_NAME=$1
AITER_DIR=$2
ARCH=$3
CUSTOM_BRANCH=$4
CUSTOM_AITER_PATCHES=$5

echo -e "${GREEN}Set Aiter repository ${NC}"

# Pull the specified image and retrieve the vllm version
docker pull "$DOCKER_IMAGE_NAME"
AITER_COMMIT=$(get_aiter_version_hash "$DOCKER_IMAGE_NAME")

if [ -n "$CUSTOM_BRANCH" ]; then
    AITER_COMMIT="$CUSTOM_BRANCH"
fi

echo -e "${GREEN}aiter git commit: ${AITER_COMMIT} ${NC}"

if [ -d "$AITER_DIR" ]; then
    echo -e "${YELLOW}Remove existing aiter dir: ${AITER_DIR} ${NC}"
    rm -rf $AITER_DIR
fi
# TODO This avoids aiter re-build, avoid this later
if [ "$ARCH" == "gfx950" ]; then
    AITER_CONTAINER_NAME="get-aiter-folder"
    docker run --name $AITER_CONTAINER_NAME -d $DOCKER_IMAGE_NAME
    docker cp $AITER_CONTAINER_NAME:/app/aiter "$AITER_DIR"
    docker stop $AITER_CONTAINER_NAME
    docker rm $AITER_CONTAINER_NAME
else
    git clone --filter=blob:none --recursive https://github.com/ROCm/aiter.git "$AITER_DIR"
fi

# Make it absolute to work properly with subsequent script calls
AITER_DIR=$(readlink -e $AITER_DIR)

# Switch to the commit
git -C "$AITER_DIR" checkout $AITER_COMMIT

cd "$SCRIPT_DIR/aiter/"

git submodule sync && git submodule update --init --recursive

PATCH_FILE_FOLDER="$SCRIPT_DIR/aiter_patches"

if [ -n "$CUSTOM_AITER_PATCHES" ]; then
    echo -e "${GREEN}Apply patches.. ${NC}"

    PATCH_LIST_FILE="$PATCH_FILE_FOLDER/patch_files_${ARCH}.txt"
    apply_patches $PATCH_LIST_FILE $PATCH_FILE_FOLDER

    echo -e "${GREEN}All patches applied successfully${NC}"
fi
