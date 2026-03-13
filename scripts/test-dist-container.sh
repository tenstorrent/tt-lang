#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Test the latest dist Docker container on a machine with hardware.
#
# The dist image ships a fully pre-built tt-lang with an auto-activated
# environment. This script verifies that examples and tutorials run
# correctly out of the box.
#
# Usage:
#   scripts/test-dist-container.sh [OPTIONS]
#
# Options:
#   --tag TAG       Docker image tag (default: latest)
#   --skip-pull     Skip docker pull (use local image)
#   --shell         Drop into a shell instead of running tests
#
# Prerequisites:
#   - Docker installed and accessible via sudo
#   - Tenstorrent device at /dev/tenstorrent/0
#   - Hugepages configured

set -euo pipefail

: "${TT_CARD_NUM:=0}"
TAG="latest"
SKIP_PULL=false
SHELL_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)       TAG="$2"; shift 2 ;;
        --skip-pull) SKIP_PULL=true; shift ;;
        --shell)     SHELL_MODE=true; shift ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

IMAGE="ghcr.io/tenstorrent/tt-lang/tt-lang-dist-ubuntu-22-04:${TAG}"
CONTAINER="ttlang-dist-test-$$"

echo "=== Dist Container Test ==="
echo "  Image:     $IMAGE"
echo "  Container: $CONTAINER"
echo ""

# Pull image
if [ "$SKIP_PULL" = false ]; then
    echo "Pulling image..."
    sudo docker pull "$IMAGE"
    echo ""
fi

# Test script that runs inside the container.
# The entrypoint auto-activates the environment, so python/ttl/ttnn are
# available immediately.
TEST_SCRIPT=$(cat <<'INNER_EOF'
#!/bin/bash
set -euo pipefail

echo "=== Inside dist container ==="
echo "  Python:    $(which python3)"
echo "  Toolchain: ${TTLANG_TOOLCHAIN_DIR:-not set}"
echo ""

# Verify environment is activated
echo "=== Verifying environment ==="
python3 -c "import ttnn; print(f'  ttnn: {ttnn.__file__}')"
python3 -c "import ttl; print(f'  ttl:  {ttl.__file__}')"
echo ""

# Run tutorial examples
echo "=== Running tutorial examples ==="
TUTORIALS=(
    examples/tutorial/single_core_single_tile_block.py
    examples/tutorial/single_core_multitile_block.py
    examples/tutorial/single_core_broadcast_single_tile_block.py
    examples/tutorial/single_core_broadcast_multitile_blocks.py
    examples/tutorial/multicore.py
    examples/tutorial/multicore_grid_auto.py
)

PASSED=0
FAILED=0
for t in "${TUTORIALS[@]}"; do
    name=$(basename "$t")
    echo -n "  $name ... "
    if python3 "$t" > /tmp/test_output.log 2>&1; then
        echo "OK"
        PASSED=$((PASSED + 1))
    else
        echo "FAILED"
        FAILED=$((FAILED + 1))
        tail -20 /tmp/test_output.log | sed 's/^/    /'
    fi
done

# Run a selection of standalone examples
echo ""
echo "=== Running examples ==="
EXAMPLES=(
    examples/eltwise_add.py
    examples/singlecore_matmul.py
)

for e in "${EXAMPLES[@]}"; do
    name=$(basename "$e")
    echo -n "  $name ... "
    if python3 "$e" > /tmp/test_output.log 2>&1; then
        echo "OK"
        PASSED=$((PASSED + 1))
    else
        echo "FAILED"
        FAILED=$((FAILED + 1))
        tail -20 /tmp/test_output.log | sed 's/^/    /'
    fi
done

echo ""
echo "=== Results: $PASSED passed, $FAILED failed ==="

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
INNER_EOF
)

DOCKER_ARGS=(
    --rm
    --name "$CONTAINER"
    --device=/dev/tenstorrent/$TT_CARD_NUM:/dev/tenstorrent/0
    -v /dev/hugepages:/dev/hugepages
    -v /dev/hugepages-1G:/dev/hugepages-1G
    -e PYTHONDONTWRITEBYTECODE=1
)

if [ "$SHELL_MODE" = true ]; then
    echo "Dropping into shell. Environment is auto-activated."
    echo "Try: python3 examples/tutorial/single_core_single_tile_block.py"
    echo ""
    sudo docker run -it "${DOCKER_ARGS[@]}" "$IMAGE" bash
else
    sudo docker run "${DOCKER_ARGS[@]}" "$IMAGE" bash -c "$TEST_SCRIPT"
fi
