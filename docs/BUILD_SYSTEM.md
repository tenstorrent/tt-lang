# tt-lang Build System Documentation

## Overview

tt-lang uses a CMake-based build system that **reuses tt-mlir's environment and toolchain**. This approach:
- Avoids duplicating LLVM/MLIR builds (saves hours of build time and gigabytes of disk space)
- Ensures consistency between tt-lang and tt-mlir (same compiler, same LLVM version, same Python environment)
- Simplifies maintenance and reduces configuration complexity

The build system supports:
- C++ code for MLIR dialects and passes
- Python bindings using nanobind
- tt-mlir as a dependency (links against tt-mlir libraries and uses its Python bindings)

## Prerequisites and Build Instructions

See the [README Quick Start](README.md#quick-start) section for prerequisites and build instructions.

## How It Works

The `ExternTTMLIR.cmake` module finds your pre-built tt-mlir installation:

1. Reads the reference version from `third-party/tt-mlir.commit`
2. Looks for pre-built tt-mlir via `$TT_MLIR_HOME` or `$TTMLIR_TOOLCHAIN_DIR`
3. Links against the found tt-mlir libraries

**Note:** The `third-party/tt-mlir.commit` file contains a reference tt-mlir version for compatibility. Ensure your installed tt-mlir is compatible.

## Directory Structure

```
tt-lang/
├── CMakeLists.txt                 # Root build file
├── BUILD_SYSTEM.md                # This file
├── README.md                      # Project README
├── requirements.txt               # Python runtime requirements
├── dev-requirements.txt           # Development requirements
├── cmake/
│   └── modules/                   # CMake helper modules
│       ├── CompilerSetup.cmake
│       └── ExternTTMLIR.cmake     # tt-mlir dependency management
├── env/
│   └── activate                   # Environment activation (sources tt-mlir's env)
├── sim/
│   └── cbsim/                     # Simulator code (Python)
├── include/
│   ├── CMakeLists.txt
│   ├── ttlang/                    # Public C++ headers
│   │   └── Dialect/TTMetal/Pipelines/
│   └── ttlang-c/                  # Public C API headers
├── lib/
│   ├── CMakeLists.txt
│   └── Dialect/TTMetal/Pipelines/ # TTMetal pipeline implementations
├── python/
│   ├── CMakeLists.txt             # Python bindings build
│   ├── pyproject.toml             # Python project configuration
│   ├── setup.py                   # Python package setup
│   └── ttlang/                    # Python package
│       └── __init__.py
├── third-party/
│   ├── CMakeLists.txt
│   └── tt-mlir.commit             # Reference tt-mlir version
├── tools/
│   └── ttlang-opt/                # Command-line tool
├── test/
│   ├── CMakeLists.txt             # Lit test configuration
│   ├── pytest.ini                 # Pytest configuration for tests
│   └── sim/
│       └── test_cbsim.py          # Simulator tests
└── tests/                         # (empty placeholder)
```

## Build Process

### 1. Activate Environment

The tt-lang environment sources tt-mlir's environment and adds tt-lang-specific paths:

```bash
cd /path/to/tt-lang
source env/activate
```

**What this does:**
- Sets `TT_MLIR_HOME` (auto-detects if tt-mlir is in `../tt-mlir` relative to tt-lang)
- Sources tt-mlir's `env/activate` (sets up `TTMLIR_TOOLCHAIN_DIR`, `TTMLIR_VENV_DIR`, Python venv, etc.)
- Sets `TT_LANG_HOME` to tt-lang project root
- Sets `TTLANG_ENV_ACTIVATED=1`
- Prepends `tt-lang/build/bin` to PATH
- Prepends `tt-lang/build/python_packages` to PYTHONPATH

**Custom tt-mlir location:**
If tt-mlir is not in the default location, set `TT_MLIR_HOME`:
```bash
export TT_MLIR_HOME=/path/to/tt-mlir
source env/activate
```

### 2. Configure tt-lang

```bash
cmake -GNinja -Bbuild .
```

**Configuration Options:**

- `CMAKE_BUILD_TYPE` (default: Release) - Build type (Debug, Release, RelWithDebInfo, Asan, Coverage, Assert)
- `TTLANG_ENABLE_BINDINGS_PYTHON` (default: ON) - Enable Python bindings
- `TTLANG_ENABLE_RUNTIME` (default: OFF) - Enable runtime support
- `CODE_COVERAGE` (default: OFF) - Enable code coverage reporting

**Example - Debug build:**
```bash
cmake -GNinja -Bbuild . -DCMAKE_BUILD_TYPE=Debug
```

### 3. Build tt-lang

```bash
cmake --build build
```

This builds:
- C++ libraries (when added to `lib/`)
- Python bindings (in `build/python_packages/ttlang/`)

### 4. Use Python Packages

After building, Python packages are available via PYTHONPATH:

```bash
# Already set by env/activate
python3 -c "import ttlang; print(ttlang.__version__)"
```

Or install as an editable package:
```bash
pip install -e python/
```

## Integration with tt-mlir

### Using tt-mlir's Toolchain

tt-lang **reuses everything from tt-mlir**:
- LLVM/MLIR installation at `$TTMLIR_TOOLCHAIN_DIR`
- Python virtual environment at `$TTMLIR_VENV_DIR`
- CMake modules from `$TT_MLIR_HOME/cmake/modules`
- Build tools (clang, ninja, llvm-lit, etc.)

### Using tt-mlir's CMake Modules

tt-lang uses tt-mlir's CMake helper modules:
- `FindMLIR.cmake` - Locates MLIR/LLVM
- `TTMLIRBuildTypes.cmake` - Custom build types (Asan, Coverage, Assert)
- Others as needed

These are automatically available via:
```cmake
list(APPEND CMAKE_MODULE_PATH "$ENV{TT_MLIR_HOME}/cmake/modules")
```

### Linking Against tt-mlir

tt-mlir provides `TTMLIRConfig.cmake` that exports targets and variables:

**Variables:**
- `TTMLIR_INCLUDE_DIRS` - Include directories
- `TTMLIR_LIBRARY_DIRS` - Library directories
- `TTMLIR_CMAKE_DIR` - CMake configuration directory

**Targets:**
- `MLIRTTCoreDialect`
- `MLIRTTNNDialect`
- `MLIRTTIRDialect`
- `TTMLIRSupport`
- Many more...

**Usage in CMakeLists.txt:**
```cmake
# Link against tt-mlir libraries
target_link_libraries(MyTarget
  PRIVATE
    MLIRTTCoreDialect
    MLIRTTNNDialect
    # other tt-mlir targets
)
```

## Environment Variables

### Set by tt-mlir's environment:
- `TTMLIR_TOOLCHAIN_DIR` - Toolchain installation directory (e.g., `/opt/ttmlir-toolchain`)
- `TTMLIR_VENV_DIR` - Python virtual environment directory
- `TTMLIR_ENV_ACTIVATED` - Set to 1 when tt-mlir environment is active
- `TT_MLIR_HOME` - tt-mlir project root

### Set by tt-lang's environment:
- `TT_LANG_HOME` - tt-lang project root
- `TTLANG_ENV_ACTIVATED` - Set to 1 when tt-lang environment is active

### Modified by tt-lang:
- `PATH` - Prepends `$TT_LANG_HOME/build/bin`
- `PYTHONPATH` - Prepends `$TT_LANG_HOME/build/python_packages`

## Development Workflow

### First-time Setup

```bash
# See Prerequisites section for tt-mlir setup

# Build tt-lang
cd /path/to/tt-lang
source env/activate  # This sources tt-mlir's env automatically
cmake -GNinja -Bbuild .  # Will find or fetch tt-mlir
cmake --build build
```

### Daily Development

```bash
# In each new shell session:
cd /path/to/tt-lang
source env/activate

# Build
cmake --build build

# Test
pytest tests/
```

### Rebuilding After Changes

```bash
# After CMakeLists.txt or other CMake changes:
cmake -GNinja -Bbuild .

# After C++ code changes:
cmake --build build

# After Python code changes (if no C++ changes):
# Python files are used directly from source or build/python_packages
```


## Future Additions

As the project grows, you can add:

### Dialects
- Add headers to `include/ttlang/Dialect/`
- Add implementation to `lib/Dialect/`
- Add TableGen definitions (`.td` files)

### Passes and Transformations
- Add headers to `include/ttlang/Transforms/`
- Add implementation to `lib/Transforms/`

### C API
- Add headers to `include/ttlang-c/`
- Add implementation to `lib/CAPI/`

### Command-line Tools
- Create tool directories (e.g., `tools/ttlang-opt/`, `tools/ttlang-translate/`)
- Add executables that link against tt-lang and tt-mlir libraries

### Python Extensions
- Add C++ Python binding files in `python/` (e.g., `TTLangModule.cpp`)
- Declare in `python/CMakeLists.txt` using `declare_mlir_python_extension`

### Tests
- Add lit tests in `test/` directory (TODO: add more details)
- Add unit tests for C++ code
- Add Python tests in `tests/`

## Troubleshooting

### Error: "TTLANG_ENV_ACTIVATED not set"
**Solution:** Run `source env/activate`

### Error: "TTMLIR_ENV_ACTIVATED not set"
**Solution:** Ensure tt-mlir environment is properly sourced. The tt-lang activate script should do this automatically, but verify `TT_MLIR_HOME` is set correctly.

### Error: "TT_MLIR_HOME not set and tt-mlir not found"
**Solution:** Set `TT_MLIR_HOME` before activating:
```bash
export TT_MLIR_HOME=/path/to/tt-mlir
source env/activate
```

### Error: "Could not find TTMLIR"
**Solution:** Ensure tt-mlir is built and installed:
```bash
cd $TT_MLIR_HOME
source env/activate
cmake --build build
```

The `TTMLIRConfig.cmake` should be at one of:
- `$TTMLIR_TOOLCHAIN_DIR/lib/cmake/ttmlir/TTMLIRConfig.cmake` (if installed)
- `$TT_MLIR_HOME/build/lib/cmake/ttmlir/TTMLIRConfig.cmake` (build tree)

### Python import errors
**Solution:** Ensure environment is activated and paths are correct:
```bash
source env/activate
echo $PYTHONPATH  # Should include both tt-mlir and tt-lang packages
python3 -c "import ttmlir; import ttlang"
```

### Build errors about missing LLVM/MLIR
**Solution:** Ensure tt-mlir's toolchain is built:
```bash
cd $TT_MLIR_HOME
cmake -GNinja -Bbuild-env env/
cmake --build build-env
```
