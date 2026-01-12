#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Setup script for TT-Lang simulator environment.
#
# Creates a Python virtual environment and installs dependencies needed
# to run the simulator without building the full compiler stack.
#
# Usage:
#   ./setup_simulator.sh
#
# After setup completes:
#   source .venv/bin/activate
#   ./bin/ttlsim examples/eltwise_add.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"

# Warn if already in a virtual environment
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "Error: Already in a virtual environment: $VIRTUAL_ENV"
    echo "Please deactivate first by running: deactivate"
    exit 1
fi

echo "========================================="
echo "TT-Lang Simulator Setup"
echo "========================================="
echo ""

# Find Python 3.11+ automatically
echo "Looking for Python 3.11+..."
PYTHON=""

# Try common Python 3.11+ executable names
for py in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$py" &> /dev/null; then
        PY_VERSION=$($py -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null || echo "0.0")
        PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

        if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 11 ]; then
            PYTHON="$py"
            PYTHON_VERSION="$PY_VERSION"
            echo "✓ Found $py (Python $PYTHON_VERSION)"
            break
        fi
    fi
done

# If still no Python found, error out
if [ -z "$PYTHON" ]; then
    echo "Error: Could not find Python 3.11+ on your system"
    echo ""
    echo "To install Python 3.11+ on macOS:"
    echo "  brew install python@3.11"
    echo ""
    echo "Or set PYTHON environment variable:"
    echo "  PYTHON=/path/to/python3.11 ./setup_simulator.sh"
    exit 1
fi
echo ""

# Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    read -p "Remove and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    "$PYTHON" -m venv "$VENV_DIR"
    echo "✓ Virtual environment created"
    echo ""
fi

# Verify venv Python version
VENV_VERSION=$("$VENV_DIR/bin/python" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
echo "Virtual environment Python: $VENV_VERSION"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing runtime dependencies..."
pip install -r requirements.txt --quiet
echo "✓ Runtime dependencies installed"
echo ""

echo "Installing development dependencies..."
pip install -r dev-requirements.txt --quiet
echo "✓ Development dependencies installed"
echo ""

# Configure PYTHONPATH in activation script
echo "Configuring PYTHONPATH..."
REPO_PYTHON_DIR="$SCRIPT_DIR/python"
ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"

# Remove any existing PYTHONPATH configuration
sed -i.bak '/# TT-Lang PYTHONPATH/,/# End TT-Lang PYTHONPATH/d' "$ACTIVATE_SCRIPT" 2>/dev/null || true

# Add PYTHONPATH configuration
cat >> "$ACTIVATE_SCRIPT" << EOF

# TT-Lang PYTHONPATH configuration
export TTLANG_OLD_PYTHONPATH="\${PYTHONPATH:-}"
export PYTHONPATH="$REPO_PYTHON_DIR:\$PYTHONPATH"
# End TT-Lang PYTHONPATH configuration
EOF

# Update deactivate function to restore PYTHONPATH
if ! grep -q "TTLANG_OLD_PYTHONPATH" "$ACTIVATE_SCRIPT" | grep -q "deactivate"; then
    sed -i.bak '/^deactivate () {/a\
    # Restore old PYTHONPATH\
    if [ -n "${TTLANG_OLD_PYTHONPATH+_}" ]; then\
        export PYTHONPATH="$TTLANG_OLD_PYTHONPATH"\
        unset TTLANG_OLD_PYTHONPATH\
    fi
' "$ACTIVATE_SCRIPT"
fi

# Apply PYTHONPATH to current session
export PYTHONPATH="$REPO_PYTHON_DIR:${PYTHONPATH:-}"

echo "✓ PYTHONPATH configured"
echo ""

# Clean up backup files
rm -f "$ACTIVATE_SCRIPT.bak"

echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "PYTHONPATH will be automatically configured to include:"
echo "  - $REPO_PYTHON_DIR"
echo ""
echo "Then you can run simulator examples:"
echo "  ./bin/ttlsim examples/eltwise_add.py"
echo ""
echo "To run simulator tests:"
echo "  pytest test/sim/"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
echo ""
