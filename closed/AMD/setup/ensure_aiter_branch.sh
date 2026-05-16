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

function get_aiter_version_hash() {
    local IMAGE_NAME="$1"

    local LINE_COMMIT_ID
    LINE_COMMIT_ID=$(
        apptainer_exec_image "$IMAGE_NAME" python3 -m pip list \
            | grep '^\baiter\b' \
            | awk '{print $2}' || true
    )

    local REGEX_COMMIT_ID=".*\+g([A-Za-z0-9]+).*"

    if [[ "$LINE_COMMIT_ID" =~ $REGEX_COMMIT_ID ]]; then
        echo "${BASH_REMATCH[1]}"
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

    if [[ ! -f "$patch_list_file" ]]; then
        echo -e "${RED}Patch list file '$patch_list_file' not found.${NC}"
        exit 1
    fi

    while IFS= read -r line || [ -n "$line" ]; do
        line=$(echo "$line" | sed 's/[[:space:]]*$//')

        if [[ -z "$line" || "$line" == \#* ]]; then
            echo -e "${YELLOW}Patch file skipped: $line ${NC}"
            continue
        fi

        module="${line%% *}"
        patch_file="${line#* }"

        if [[ -z "${module_path[$module]+exists}" ]]; then
            echo -e "${RED}Unknown module '$module' in patch list '$patch_list_file'.${NC}"
            exit 1
        fi

        pushd "${module_path[$module]}" > /dev/null

        if [[ ! -f "$patch_folder/$patch_file" ]]; then
            echo -e "${RED}Patch file '$patch_folder/$patch_file' not found. ${NC}"
            exit 1
        fi

        echo -e "${GREEN}Applying patch '$module' '$patch_folder/$patch_file'... ${NC}"
        git apply "$patch_folder/$patch_file"

        popd > /dev/null
    done < "$patch_list_file"
}

if [[ $# -lt 5 ]]; then
    echo -e "${RED}Usage: $0 <BASE_IMAGE_NAME> <AITER_DIR> <ARCH> <CUSTOM_BRANCH> <CUSTOM_AITER_PATCHES>${NC}"
    exit 1
fi

BASE_IMAGE_NAME="$1"
AITER_DIR="$2"
ARCH="$3"
CUSTOM_BRANCH="$4"
CUSTOM_AITER_PATCHES="$5"

echo -e "${GREEN}Set Aiter repository ${NC}"

if [[ -n "$CUSTOM_BRANCH" ]]; then
    AITER_COMMIT="$CUSTOM_BRANCH"
else
    AITER_COMMIT=$(get_aiter_version_hash "$BASE_IMAGE_NAME")
fi

if [[ -z "${AITER_COMMIT:-}" ]]; then
    echo -e "${RED}Could not detect aiter git commit from image: $BASE_IMAGE_NAME${NC}"
    echo -e "${RED}If you need aiter, pass --custom-aiter-branch <commit-or-branch>. For gfx1201, this is usually not needed.${NC}"
    exit 1
fi

echo -e "${GREEN}aiter git commit: ${AITER_COMMIT} ${NC}"

if [[ -d "$AITER_DIR" ]]; then
    echo -e "${YELLOW}Remove existing aiter dir: ${AITER_DIR} ${NC}"
    rm -rf "$AITER_DIR"
fi

if [[ "$ARCH" == "gfx950" ]]; then
    echo -e "${GREEN}Copy /app/aiter from base image using Apptainer ${NC}"

    mkdir -p "$(dirname "$AITER_DIR")"

    apptainer_exec_image "$BASE_IMAGE_NAME" \
        bash -lc "cd /app && tar -cf - aiter" \
        | tar -xf - -C "$(dirname "$AITER_DIR")"

    if [[ "$(readlink -f "$(dirname "$AITER_DIR")/aiter")" != "$(readlink -f "$AITER_DIR" 2>/dev/null || true)" ]]; then
        rm -rf "$AITER_DIR"
        mv "$(dirname "$AITER_DIR")/aiter" "$AITER_DIR"
    fi
else
    git clone --filter=blob:none --recursive https://github.com/ROCm/aiter.git "$AITER_DIR"
fi

AITER_DIR=$(readlink -e "$AITER_DIR")

git -C "$AITER_DIR" checkout "$AITER_COMMIT"

cd "$AITER_DIR"

git submodule sync
git submodule update --init --recursive

PATCH_FILE_FOLDER="$SCRIPT_DIR/aiter_patches"

if [[ -n "$CUSTOM_AITER_PATCHES" ]]; then
    echo -e "${GREEN}Apply patches.. ${NC}"

    PATCH_LIST_FILE="$PATCH_FILE_FOLDER/patch_files_${ARCH}.txt"
    apply_patches "$PATCH_LIST_FILE" "$PATCH_FILE_FOLDER"

    echo -e "${GREEN}All patches applied successfully${NC}"
fi