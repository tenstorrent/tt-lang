#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Test the latest ird Docker container on a machine with hardware.
#
# Clones tt-lang inside the container, builds with the pre-installed
# toolchain, and runs the full test suite.
#
# Usage:
#   scripts/test-ird-container.sh [OPTIONS]
#
# Options:
#   --tag TAG       Docker image tag (default: latest)
#   --branch REF    tt-lang branch/SHA to test (default: main)
#   --skip-pull     Skip docker pull (use local image)
#   --shell         Drop into a shell instead of running tests
#   --card N        Tenstorrent card number (default: 0, or TT_CARD_NUM env)
#
# Prerequisites:
#   - Docker installed and accessible via sudo
#   - Tenstorrent device available
#   - Hugepages configured

set -euo pipefail

: "${TT_CARD_NUM:=0}"
TAG="latest"
BRANCH="main"
SKIP_PULL=false
SHELL_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --card)      TT_CARD_NUM="$2"; shift 2 ;;
        --tag)       TAG="$2"; shift 2 ;;
        --branch)    BRANCH="$2"; shift 2 ;;
        --skip-pull) SKIP_PULL=true; shift ;;
        --shell)     SHELL_MODE=true; shift ;;
        --help)      echo "Usage: $0 [OPTIONS]"; exit 0 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

IMAGE="ghcr.io/tenstorrent/tt-lang/tt-lang-ird-ubuntu-22-04:${TAG}"
CONTAINER="ttlang-ird-test-$$"

echo "=== IRD Container Test ==="
echo "  Image:     $IMAGE"
echo "  Branch:    $BRANCH"
echo "  Card:      $TT_CARD_NUM"
echo "  Container: $CONTAINER"
echo ""

# Pull image
if [ "$SKIP_PULL" = false ]; then
    echo "Pulling image..."
    sudo docker pull "$IMAGE"
    echo ""
fi

TEST_SCRIPT=$(cat <<'INNER_EOF'
#!/bin/bash
set -euo pipefail

# Temporarily install newer git (for --filter support) and GNU time
# TODO: remove once base image includes ppa:git-core/ppa and time
apt-get update -qq > /dev/null 2>&1
add-apt-repository ppa:git-core/ppa -y > /dev/null 2>&1
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq git time > /dev/null 2>&1

echo "=== Inside container ==="
echo "  Toolchain: $TTLANG_TOOLCHAIN_DIR"
echo "  Python:    $(which python3)"
echo "  Git:       $(git --version)"
echo ""

mkdir -p /workspace && cd /workspace
git config --global --add safe.directory '*'

# Use GNU time for wall-clock + peak memory reporting
TIME="/usr/bin/time -v"

# Clone tt-lang (blobless clone for speed — skips file content until needed)
echo "=== Cloning tt-lang (branch: $BRANCH) ==="
$TIME git clone --depth 1 --filter=blob:none --branch "$BRANCH" \
    https://github.com/tenstorrent/tt-lang.git
cd tt-lang

# Configure (submodules are auto-initialized by CMake during configure)
echo ""
echo "=== Configuring tt-lang ==="
$TIME cmake -G Ninja -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DTTLANG_USE_TOOLCHAIN=ON

# Build
echo ""
echo "=== Building tt-lang ==="
source build/env/activate
$TIME cmake --build build

# Install test deps
echo ""
echo "=== Installing test dependencies ==="
pip install -r dev-requirements.txt --quiet

# Run tests
echo ""
echo "=== Running smoketest ==="
python3 test/python/smoketest.py

echo ""
echo "=== Running simple_add.py ==="
python test/python/simple_add.py

echo ""
echo "=== Running all compiler tests ==="
cmake --build build --target check-ttlang-all

echo ""
echo "=== Running Simulator tests ==="
python3 -m pytest test/sim -v --tb=short

echo ""
echo "=== All tests passed ==="
INNER_EOF
)

DOCKER_ARGS=(
    --rm
    --name "$CONTAINER"
    --device=/dev/tenstorrent/$TT_CARD_NUM:/dev/tenstorrent/0
    -v /dev/hugepages:/dev/hugepages
    -v /dev/hugepages-1G:/dev/hugepages-1G
    -e BRANCH="$BRANCH"
    -e PYTHONDONTWRITEBYTECODE=1
    -e TTLANG_TEST_SEED=42
)

if [ "$SHELL_MODE" = true ]; then
    echo "Dropping into shell. To build and test:"
    echo ""
    echo "  cd /workspace"
    echo "  git clone --depth 1 --filter=blob:none --branch $BRANCH https://github.com/tenstorrent/tt-lang.git"
    echo "  cd tt-lang"
    echo "  git submodule update --init --depth 1 third-party/tt-mlir third-party/tt-metal"
    echo "  cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DTTLANG_USE_TOOLCHAIN=ON"
    echo "  source build/env/activate"
    echo "  cmake --build build"
    echo ""
    sudo docker run -it "${DOCKER_ARGS[@]}" "$IMAGE" bash
else
    sudo docker run "${DOCKER_ARGS[@]}" "$IMAGE" bash -c "$TEST_SCRIPT"
fi
