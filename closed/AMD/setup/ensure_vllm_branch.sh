#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
APPTAINER_EXEC_OPTS="${APPTAINER_EXEC_OPTS:-}"

function image_ref_for_apptainer() {
    local IMAGE_NAME="$1"

    if [[ -f "$IMAGE_NAME" ]]; then
        readlink -f "$IMAGE_NAME"
        return
    fi

    if [[ -f "${IMAGE_NAME}.sif" ]]; then
        readlink -f "${IMAGE_NAME}.sif"
        return
    fi

    echo "docker://${IMAGE_NAME}"
}

function apptainer_exec_image() {
    local IMAGE_NAME="$1"
    shift

    local IMAGE_REF
    IMAGE_REF=$(image_ref_for_apptainer "$IMAGE_NAME")

    "$APPTAINER_BIN" exec \
        $APPTAINER_EXEC_OPTS \
        "$IMAGE_REF" \
        "$@"
}

function get_vllm_version_hash() {
    local IMAGE_NAME="$1"

    local LINE_COMMIT_ID
    LINE_COMMIT_ID=$(
        apptainer_exec_image "$IMAGE_NAME" python3 -m pip list \
            | grep '^\bvllm\b' \
            | awk '{print $2}' || true
    )

    local REGEX_COMMIT_ID=".*\+g([A-Za-z0-9]+).*"

    if [[ "$LINE_COMMIT_ID" =~ $REGEX_COMMIT_ID ]]; then
        echo "${BASH_REMATCH[1]}"
    fi
}

function apply_patches() {
    local patch_list_file="$1"
    local patch_folder="$2"

    if [[ ! -f "$patch_list_file" ]]; then
        echo -e "${RED}Patch list file '$patch_list_file' not found.${NC}"
        exit 1
    fi

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
    done < "$patch_list_file"
}

if [[ $# -lt 7 ]]; then
    echo -e "${RED}Usage: $0 <BASE_IMAGE_NAME> <VLLM_DIR> <ARCH> <CUSTOM_BRANCH> <CUSTOM_VLLM_PATCHES> <CUSTOM_LLAMA2_PATCHES> <CUSTOM_MOE_PATCHES>${NC}"
    exit 1
fi

BASE_IMAGE_NAME="$1"
VLLM_DIR="$2"
ARCH="$3"
CUSTOM_BRANCH="$4"
CUSTOM_VLLM_PATCHES="$5"
CUSTOM_LLAMA2_PATCHES="$6"
CUSTOM_MOE_PATCHES="$7"

echo -e "${GREEN}Set VLLM repository ${NC}"

if [[ -n "$CUSTOM_BRANCH" ]]; then
    VLLM_COMMIT="$CUSTOM_BRANCH"
else
    VLLM_COMMIT=$(get_vllm_version_hash "$BASE_IMAGE_NAME")
fi

if [[ -z "${VLLM_COMMIT:-}" ]]; then
    echo -e "${RED}Could not detect vLLM git commit from image: $BASE_IMAGE_NAME${NC}"
    echo -e "${RED}Please pass --custom-vllm-branch <commit-or-branch>. For your successful build log, use 71faa1880.${NC}"
    exit 1
fi

echo -e "${GREEN}VLLM git commit: ${VLLM_COMMIT} ${NC}"

if [[ -d "$VLLM_DIR" ]]; then
    echo -e "${YELLOW}Remove existing vllm dir: ${VLLM_DIR} ${NC}"
    rm -rf "$VLLM_DIR"
fi

if [[ "$ARCH" == "gfx950" ]]; then
    echo -e "${GREEN}Copy /app/vllm from base image using Apptainer ${NC}"

    mkdir -p "$(dirname "$VLLM_DIR")"

    apptainer_exec_image "$BASE_IMAGE_NAME" \
        bash -lc "cd /app && tar -cf - vllm" \
        | tar -xf - -C "$(dirname "$VLLM_DIR")"

    if [[ "$(readlink -f "$(dirname "$VLLM_DIR")/vllm")" != "$(readlink -f "$VLLM_DIR" 2>/dev/null || true)" ]]; then
        rm -rf "$VLLM_DIR"
        mv "$(dirname "$VLLM_DIR")/vllm" "$VLLM_DIR"
    fi

    rm -rf "$VLLM_DIR"/build
else
    git clone --filter=blob:none https://github.com/ROCm/vllm.git "$VLLM_DIR"
fi

VLLM_DIR=$(readlink -e "$VLLM_DIR")

git -C "$VLLM_DIR" checkout "$VLLM_COMMIT"

cd "$VLLM_DIR"

PATCH_FILE_FOLDER="$SCRIPT_DIR/vllm_patches"

if [[ -n "$CUSTOM_VLLM_PATCHES" ]]; then
    echo -e "${GREEN}Apply common patches.. ${NC}"
    COMMON_PATCH_FILES="$PATCH_FILE_FOLDER/common_patch_files.txt"
    apply_patches "$COMMON_PATCH_FILES" "$PATCH_FILE_FOLDER"
fi

if [[ -n "$CUSTOM_LLAMA2_PATCHES" ]]; then
    echo -e "${GREEN}Apply llama2 patches.. ${NC}"
    LLAMA2_PATCH_FILES="$PATCH_FILE_FOLDER/llama2_patch_files_${ARCH}.txt"
    apply_patches "$LLAMA2_PATCH_FILES" "$PATCH_FILE_FOLDER"
fi

if [[ -n "$CUSTOM_MOE_PATCHES" ]]; then
    echo -e "${GREEN}Apply moe patches.. ${NC}"
    MOE_PATCH_FILES="$PATCH_FILE_FOLDER/moe_patch_files.txt"
    apply_patches "$MOE_PATCH_FILES" "$PATCH_FILE_FOLDER"
fi

echo -e "${GREEN}All patches applied successfully ${NC}"