#!/bin/bash
set -e

SETUP_DIR=$(dirname -- $0)
VLLM_IMAGE_NAME=rocm/vllm-dev:nightly_aiter_integration_final_20250325_634

if ! docker image inspect "${VLLM_IMAGE_NAME}" > /dev/null 2>&1; then
rm -rf $SETUP_DIR/vllm
git clone https://github.com/rocm/vllm/ $SETUP_DIR/vllm --recursive && cd $SETUP_DIR/vllm
git switch aiter_integration_final
sed -i -e "s/\bsetuptools\b/'setuptools<80'/g" -e "s/\bpybind11\b/'pybind11<3'/g" Dockerfile.rocm_base
sed -i -e "s/\btransformers >= 4.48.2\b/transformers==4.50.1/g" requirements/common.txt
sed -i -e "s/\bray >= 2.10.0\b/ray>=2.10.0,<2.45.0/g" requirements/rocm.txt
# Note: DOCKER_BUILDKIT requires "docker-buildx-plugin" package
DOCKER_BUILDKIT=1 docker build --build-arg BASE_IMAGE=rocm/dev-ubuntu-22.04:6.3.4-complete -f Dockerfile.rocm_base -t $VLLM_IMAGE_NAME-base .
DOCKER_BUILDKIT=1 docker build --build-arg BASE_IMAGE=$VLLM_IMAGE_NAME-base -f Dockerfile.rocm -t $VLLM_IMAGE_NAME .
cd -
rm -rf $SETUP_DIR/vllm
fi

# Create mixtral-8x7b docker
bash $SETUP_DIR/build_docker.sh --base-docker-image $VLLM_IMAGE_NAME --apply-vllm-patches --apply-moe-patches --apply-aiter-patches --custom-aiter-branch 7862472b
image_name=$(cat mlperf_image_name.txt)
if [ ! -z "$1" ]; then docker tag $image_name $1; fi
