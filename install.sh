#!/bin/bash
set -euo pipefail

# Auto-detect GPU architecture and compile only for current GPU.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    CUDA_ARCH=$(python -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')" 2>/dev/null) || {
        echo "Error: Failed to detect GPU architecture. Ensure CUDA is available."
        exit 1
    }
    export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH"
    echo "Detected GPU architecture: $CUDA_ARCH"
fi

# Install standalone lietorch shared with other baselines.
LIETORCH_PATH="../DROID_SLAM/thirdparty/lietorch"
if ! python -c "import lietorch" 2>/dev/null; then
    if [ ! -d "$LIETORCH_PATH" ]; then
        echo "Error: lietorch was not found at $LIETORCH_PATH"
        echo "Initialize submodules and run setup again."
        exit 1
    fi
    echo "Installing lietorch from $LIETORCH_PATH..."
    pip install -v "$LIETORCH_PATH" --no-build-isolation
else
    echo "lietorch already installed, skipping..."
fi

EIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR:-/usr/include/eigen3}"
if [ ! -d "$EIGEN3_INCLUDE_DIR" ]; then
    echo "Error: system Eigen3 include dir not found at '$EIGEN3_INCLUDE_DIR'."
    echo "Install Eigen3 (e.g. apt install libeigen3-dev) or set EIGEN3_INCLUDE_DIR."
    exit 1
fi

pip install -v -e . --no-build-isolation
