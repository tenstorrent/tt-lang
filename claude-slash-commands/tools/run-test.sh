#!/bin/bash
# Run a tt-lang test on a remote machine
# Usage: ./run-test.sh [-v|--verbose] [--emit-runner] <test_path> [extra_args...]
#
# By default, output is saved to /tmp/ttlang_test_output.log on the remote (silent mode).
# Use -v to stream output to terminal AND enable verbose MLIR passes.
# Use --emit-runner to generate C++ kernels and Python runner in /tmp/$USER/.
#
# Examples:
#   ./run-test.sh test/python/test_dram_interleaved_add.py   # Silent, check log after
#   ./run-test.sh -v test/python/test_dram_interleaved_add.py # Stream + verbose passes
#   ./run-test.sh --emit-runner /absolute/path/to/my_test.py  # Emit kernels + runner
#   ./run-test.sh /absolute/path/to/my_test.py
#
# Output locations on remote:
#   /tmp/ttlang_test_output.log  - Test stdout/stderr (always saved)
#   /tmp/ttlang_initial.mlir     - Initial MLIR before passes
#   /tmp/ttlang_final.mlir       - Final MLIR after passes
#
# To read log after test:
#   remote-run.sh cat /tmp/ttlang_test_output.log
#   remote-run.sh tail -200 /tmp/ttlang_test_output.log

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Parse flags
VERBOSE=""
STREAM_OUTPUT=""
EMIT_RUNNER=""
while [[ "$1" == -* ]]; do
    case "$1" in
        -v|--verbose)
            VERBOSE=1
            STREAM_OUTPUT=1
            shift
            ;;
        --emit-runner)
            EMIT_RUNNER=1
            shift
            ;;
        -h|--help)
            # Fall through to usage below
            break
            ;;
        *)
            echo "Unknown flag: $1"
            exit 1
            ;;
    esac
done

# Load config
if [ -f "$SCRIPT_DIR/remote.conf" ]; then
    source "$SCRIPT_DIR/remote.conf"
else
    echo "Error: No config file found. Copy remote.conf.example to remote.conf and configure."
    exit 1
fi

if [ $# -eq 0 ] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: $0 [-v|--verbose] [--emit-runner] <test_path> [extra_args...]"
    echo ""
    echo "Options:"
    echo "  -v, --verbose    Stream output to terminal + enable verbose MLIR passes"
    echo "                   (default: silent mode, output saved to log)"
    echo "  --emit-runner    Emit C++ kernels and Python runner to /tmp/\$USER/"
    echo ""
    echo "Examples:"
    echo "  $0 test/python/test_dram_interleaved_add.py"
    echo "  $0 -v test/python/test_dram_interleaved_add.py"
    echo "  $0 --emit-runner /Users/you/my_kernel.py"
    echo ""
    echo "Path handling:"
    echo "  /absolute/path.py  - Copied to remote automatically"
    echo "  test/python/...    - Runs as relative path on remote"
    echo ""
    echo "Output locations on remote:"
    echo "  /tmp/ttlang_test_output.log  - Test stdout/stderr"
    echo "  /tmp/ttlang_initial.mlir     - Initial MLIR"
    echo "  /tmp/ttlang_final.mlir       - Final MLIR"
    echo "  /tmp/\$USER/*.cpp            - Generated kernels (with --emit-runner)"
    echo "  /tmp/\$USER/*_runner.py      - Generated runner (with --emit-runner)"
    echo ""
    echo "To read log: remote-run.sh tail -200 /tmp/ttlang_test_output.log"
    exit 1
fi

TEST_PATH="$1"
shift
EXTRA_ARGS="$@"

# Determine if it's a local file or a path relative to tt-lang
if [[ "$TEST_PATH" = /* ]]; then
    # Absolute local path: copy to remote via stdin pipe
    if [ -f "$TEST_PATH" ]; then
        TEST_NAME=$(basename "$TEST_PATH")
        echo "Copying $TEST_NAME to remote..."
        cat "$TEST_PATH" | $REMOTE_SHELL bash -c "cat > /tmp/ttl_test_temp.py"
        REMOTE_TEST_PATH="/tmp/ttl_test_temp.py"
    else
        echo "Error: File not found: $TEST_PATH"
        exit 1
    fi
else
    # Relative path: run as-is on remote
    REMOTE_TEST_PATH="$TEST_PATH"
fi

echo "========================================"
echo "Running: $REMOTE_TEST_PATH"
if [ -n "$STREAM_OUTPUT" ]; then
    echo "Mode: verbose (streaming to terminal)"
else
    echo "Mode: silent (output saved to /tmp/ttlang_test_output.log)"
fi
echo "========================================"

# Build env vars for tt-lang test runner flags
ENV_VARS="export TTLANG_INITIAL_MLIR=/tmp/ttlang_initial.mlir;
        export TTLANG_FINAL_MLIR=/tmp/ttlang_final.mlir;"
if [ -n "$VERBOSE" ]; then
    ENV_VARS="$ENV_VARS export TTLANG_VERBOSE_PASSES=1;"
fi
if [ -n "$EMIT_RUNNER" ]; then
    ENV_VARS="$ENV_VARS export TTLANG_EMIT_RUNNER=1;"
fi

if [ -n "$STREAM_OUTPUT" ]; then
    # Verbose mode: tee to terminal AND log file
    TEST_CMD="
        exec > >(tee /tmp/ttlang_test_output.log) 2>&1
        $ENV_VARS
        python3 $REMOTE_TEST_PATH $EXTRA_ARGS
    "
    $REMOTE_SHELL bash -c "$TEST_CMD"
    EXIT_CODE=$?
else
    # Silent mode: redirect all output to log file
    TEST_CMD="
        exec > /tmp/ttlang_test_output.log 2>&1
        $ENV_VARS
        python3 $REMOTE_TEST_PATH $EXTRA_ARGS
    "
    $REMOTE_SHELL bash -c "$TEST_CMD"
    EXIT_CODE=$?
fi

echo "========================================"
if [ "$EXIT_CODE" -eq 0 ] 2>/dev/null; then
    echo "Exited with code 0 (check log for actual test results)"
else
    echo "Exited with code $EXIT_CODE"
fi
if [ -z "$STREAM_OUTPUT" ]; then
    echo ""
    echo "To view full output: remote-run.sh cat /tmp/ttlang_test_output.log"
    echo ""
    echo "Last 30 lines:"
    echo "----------------------------------------"
    $REMOTE_SHELL tail -30 /tmp/ttlang_test_output.log
fi
echo "========================================"

exit $EXIT_CODE
