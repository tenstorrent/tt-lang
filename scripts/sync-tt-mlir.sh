#!/bin/bash
# Sync tt-mlir to the version specified in third-party/tt-mlir.commit

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TT_LANG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Read the desired commit/tag
COMMIT_FILE="${TT_LANG_ROOT}/third-party/tt-mlir.commit"
if [ ! -f "$COMMIT_FILE" ]; then
    echo "ERROR: tt-mlir.commit file not found at ${COMMIT_FILE}"
    exit 1
fi

DESIRED_COMMIT=$(cat "$COMMIT_FILE" | tr -d '[:space:]')
if [ -z "$DESIRED_COMMIT" ]; then
    echo "ERROR: tt-mlir.commit file is empty"
    exit 1
fi

echo "Target tt-mlir commit: ${DESIRED_COMMIT}"

# Find TT_MLIR_HOME
if [ -z "${TT_MLIR_HOME}" ]; then
    # Try common locations
    if [ -d "/Users/bnorris/tt/tt-mlir" ]; then
        TT_MLIR_HOME="/Users/bnorris/tt/tt-mlir"
    elif [ -d "${TT_LANG_ROOT}/../tt-mlir" ]; then
        TT_MLIR_HOME="$(cd "${TT_LANG_ROOT}/../tt-mlir" && pwd)"
    else
        echo "ERROR: TT_MLIR_HOME not set and tt-mlir not found in common locations."
        echo "Please set TT_MLIR_HOME to point to your tt-mlir installation."
        exit 1
    fi
fi

if [ ! -d "$TT_MLIR_HOME" ]; then
    echo "ERROR: tt-mlir directory not found at ${TT_MLIR_HOME}"
    exit 1
fi

echo "tt-mlir location: ${TT_MLIR_HOME}"

# Check if it's a git repository
if [ ! -d "${TT_MLIR_HOME}/.git" ]; then
    echo "ERROR: ${TT_MLIR_HOME} is not a git repository"
    exit 1
fi

cd "$TT_MLIR_HOME"

# Get current commit
CURRENT_COMMIT=$(git rev-parse HEAD)
echo "Current tt-mlir commit: ${CURRENT_COMMIT}"

if [ "$CURRENT_COMMIT" = "$DESIRED_COMMIT" ]; then
    echo "✓ tt-mlir is already at the desired commit"
else
    echo "Switching tt-mlir to ${DESIRED_COMMIT}..."
    
    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        echo "WARNING: tt-mlir has uncommitted changes"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted"
            exit 1
        fi
    fi
    
    # Fetch to ensure we have the commit
    echo "Fetching from origin..."
    git fetch origin
    
    # Checkout the desired commit
    git checkout "$DESIRED_COMMIT"
    
    echo "✓ Checked out commit ${DESIRED_COMMIT}"
fi

# Rebuild tt-mlir
echo ""
echo "Rebuilding tt-mlir..."
if [ ! -f "env/activate" ]; then
    echo "ERROR: tt-mlir env/activate not found"
    exit 1
fi

source env/activate

if [ ! -d "build" ]; then
    echo "Build directory not found, running cmake configure..."
    cmake -GNinja -Bbuild .
fi

cmake --build build

echo ""
echo "✓ tt-mlir synced and built successfully"
echo ""
echo "You can now build tt-lang:"
echo "  cd ${TT_LANG_ROOT}"
echo "  source env/activate"
echo "  cmake --build build"

