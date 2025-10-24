# tt-lang environment activation (fish shell)
# This script sources tt-mlir's environment and adds tt-lang specific paths

# Step 1: Find and source tt-mlir environment
if not set -q TT_MLIR_HOME
  # Try to find tt-mlir in common locations
  if test -d "/Users/bnorris/tt/tt-mlir"
    set -gx TT_MLIR_HOME "/Users/bnorris/tt/tt-mlir"
  else if test -d "../tt-mlir"
    set -gx TT_MLIR_HOME (cd ../tt-mlir && pwd)
  else
    echo "ERROR: TT_MLIR_HOME not set and tt-mlir not found in common locations."
    echo "Please set TT_MLIR_HOME to point to your tt-mlir installation."
    echo "Example: set -gx TT_MLIR_HOME /path/to/tt-mlir"
    return 1
  end
end

if not test -f "$TT_MLIR_HOME/env/activate.fish"
  echo "ERROR: tt-mlir activate script not found at $TT_MLIR_HOME/env/activate.fish"
  echo "Please ensure TT_MLIR_HOME points to a valid tt-mlir installation."
  return 1
end

# Source tt-mlir environment (this sets up TTMLIR_TOOLCHAIN_DIR, TTMLIR_VENV_DIR, etc.)
source "$TT_MLIR_HOME/env/activate.fish"

# Step 2: Add tt-lang specific environment variables and paths
set -gx TT_LANG_HOME (pwd)

# Prepend tt-lang build directories to PATH and PYTHONPATH
set -gx PATH "$TT_LANG_HOME/build/bin" $PATH
set -gx PYTHONPATH "$TT_LANG_HOME/build/python_packages" $PYTHONPATH

echo "tt-lang environment activated (using tt-mlir from $TT_MLIR_HOME)"

