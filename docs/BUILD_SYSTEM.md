# tt-lang Build System Documentation

## Overview

tt-lang uses a CMake-based build system that **reuses tt-mlir's environment and toolchain**. This approach avoids duplicating LLVM/MLIR builds and ensures consistency between tt-lang and tt-mlir (same compiler, same LLVM version, same Python environment).

The tt-lang build system supports:
- TableGen and C++ code for MLIR dialects and passes.
- Python bindings using nanobind.
- tt-mlir as a dependency (links against tt-mlir libraries and uses its Python bindings).

## Prerequisites and Build Instructions

See the [README Quick Start](README.md#quick-start) section for prerequisites and build instructions.

## CI/CD Integration

tt-lang CI uses a **dedicated cache workflow** pattern (GitHub Actions best practice). See [CI Workflows](../.github/CI_WORKFLOWS.md) for details.

```
call-build-ttmlir-toolchain.yml (dedicated cache builder)
         |
         | builds LLVM + tt-mlir (~3-4 hours)
         v
[GitHub Actions Cache: Linux-ttlang-toolchain-v1-{sha}]
        |             |
        v             v
   CI Workflow    Container Build
   (tt-lang)      (Docker images)
```

Key points:
- Dedicated workflow (`call-build-ttmlir-toolchain.yml`) builds the toolchain
- CI and container workflows restore from cache with `fail-on-cache-miss: true`
- Cache key is based on the tt-mlir commit SHA
- Weekly schedule keeps cache warm (prevents 7-day eviction)
- Toolchain workflow triggers on changes to `third-party/tt-mlir.commit`

## Configuration and build

tt-lang supports three integration scenarios for tt-mlir:

### Scenario 1: Pre-built tt-mlir (Development Mode)

Use a tt-mlir build tree directly without installation. This mode:
- Points to a tt-mlir build directory using `TTMLIR_BUILD_DIR`
- Extracts configuration from tt-mlir's CMake cache
- Uses tt-mlir's Python environment and toolchain settings
- Does not require tt-mlir installation
- Respects `TTMLIR_TOOLCHAIN_DIR` if set as an environment variable

**Use this mode when:**
- You're actively developing tt-mlir alongside tt-lang
- You need quick iteration without rebuilding/reinstalling tt-mlir

**Configuration:**
```bash
# Set toolchain directory (optional, prevents incorrect derivation from Python path)
export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain

# Configure with tt-mlir build directory
cmake -GNinja -Bbuild . -DTTMLIR_BUILD_DIR=/path/to/tt-mlir/build
```

### Scenario 2: Pre-installed tt-mlir (Recommended)

Use a pre-installed tt-mlir toolchain. This mode:
- Finds tt-mlir at `TTMLIR_TOOLCHAIN_DIR` (default: `/opt/ttmlir-toolchain`)
- Uses the installed tt-mlir's CMake configuration
- Does not require environment activation
- Simplest and most reliable for production builds

**Use this mode when:**
- You have tt-mlir pre-installed (e.g., in a container or system installation)
- You want a stable, reproducible build environment

**Important:** The pre-installed tt-mlir must be built with Python bindings enabled (`-DTTMLIR_ENABLE_BINDINGS_PYTHON=ON`). See the [tt-mlir Getting Started guide](https://docs.tenstorrent.com/tt-mlir/getting-started.html) for details on building tt-mlir with Python bindings.

### Scenario 3: Automatic Build (FetchContent)

Automatically fetch and build tt-mlir if not found. This mode:
- Fetches tt-mlir from the commit specified in `third-party/tt-mlir.commit`
- Or uses an existing tt-mlir source directory if `TTMLIR_SRC_DIR` is provided
- Builds and installs tt-mlir locally in the build directory
- Requires an existing LLVM/MLIR toolchain and Python environment at `TTMLIR_TOOLCHAIN_DIR`
- First build is slow (~60-90 minutes), but subsequent builds reuse the cached installation

**Use this mode when:**
- You don't have tt-mlir pre-installed and don't want to build/install it yourself
- You want a fully automated setup
- You're setting up a new development environment

**Configuration:**
```bash
# Basic automatic build (fetches tt-mlir from GitHub)
cmake -GNinja -Bbuild .

# Use existing tt-mlir source directory (avoids re-downloading)
cmake -GNinja -Bbuild . -DTTMLIR_SRC_DIR=/path/to/tt-mlir-src

# Custom install prefix
cmake -GNinja -Bbuild . -DTTMLIR_INSTALL_PREFIX=/tmp/my-ttmlir-install

# With performance trace enabled
cmake -GNinja -Bbuild . -DTTLANG_ENABLE_PERF_TRACE=ON -DTTMLIR_CMAKE_BUILD_TYPE=Release
```

**Note:** CI uses `TTMLIR_SRC_DIR` to point to an already-cloned tt-mlir repository, avoiding duplicate downloads.

## How It Works

The `ExternTTMLIR.cmake` module finds tt-mlir using the following priority:

### Scenario 1: Pre-built tt-mlir
If `TTMLIR_BUILD_DIR` is specified:
1. Looks for `TTMLIRConfig.cmake` in `${TTMLIR_BUILD_DIR}/lib/cmake/ttmlir`
2. Loads configuration from tt-mlir's CMake cache using `load_cache`:
   - `CMAKE_HOME_DIRECTORY` → tt-mlir source directory
   - `_Python3_EXECUTABLE` → Python executable used by tt-mlir (typically from the virtual python environment in the tt-mlir toolchain location)
3. Sets `TTMLIR_TOOLCHAIN_DIR` from:
   - Environment variable `TTMLIR_TOOLCHAIN_DIR` (if set, takes precedence)
   - Otherwise derives it from the Python executable path
4. Adds tt-mlir's CMake modules to the module path
5. Sets up Python from `${TTMLIR_TOOLCHAIN_DIR}/venv`

### Scenario 2: Pre-installed tt-mlir
If `TTMLIR_BUILD_DIR` is not specified:
1. Sets `TTMLIR_TOOLCHAIN_DIR` from:
   - Environment variable `TTMLIR_TOOLCHAIN_DIR` (if set)
   - CMake variable `TTMLIR_TOOLCHAIN_DIR` (if set)
   - Default: `/opt/ttmlir-toolchain`
2. Looks for `TTMLIRConfig.cmake` in `${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/ttmlir`
3. If found, uses the installed tt-mlir configuration
4. Sets up Python from `${TTMLIR_TOOLCHAIN_DIR}/venv`
5. Finds MLIR and LLVM from the toolchain

### Scenario 3: Automatic build
If tt-mlir is not found in scenarios 1 or 2:
1. Reads the commit SHA from `third-party/tt-mlir.commit`
2. Uses `FetchContent_Populate` to clone tt-mlir at the above SHA
3. Configures tt-mlir with platform-specific options:
   - **Linux**: Runtime and runtime tests enabled
   - **macOS**: Runtime and runtime tests disabled
   - Common: StableHLO OFF, OPMODEL OFF, Python bindings ON, Debug strings ON
   - Performance trace: Controlled by `TTLANG_ENABLE_PERF_TRACE` (default: OFF)
4. Builds and installs tt-mlir to `${TTMLIR_INSTALL_PREFIX}` (default: `${CMAKE_BINARY_DIR}/tt-mlir-install`)
5. Uses the newly built tt-mlir for the tt-lang build

**Python Environment:**
- **Scenarios 1 & 2**: Use Python from `${TTMLIR_TOOLCHAIN_DIR}/venv` with `Python3_FIND_VIRTUALENV=ONLY`
- **Scenario 3**: Uses Python from `${TTMLIR_TOOLCHAIN_DIR}/venv` for building tt-mlir, but does not set `Python3_EXECUTABLE` globally

**Note:** The `third-party/tt-mlir.commit` file pins the exact tt-mlir version for compatibility.

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
│       ├── TTLangCompilerSetup.cmake
│       └── ExternTTMLIR.cmake     # tt-mlir dependency management
├── env/
│   └── activate.in                # Environment activation template
├── build/                         # Build directory (created by CMake)
│   └── env/
│       └── activate               # Generated activation script
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
│   ├── sim/                       # Simulator code (Python)
│   │   └── cbsim/
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

### 1. Configure

The CMake configure step generates a `build/env/activate` script tailored to your build scenario:

```bash
cd /path/to/tt-lang
cmake -GNinja -Bbuild .
```

This creates `build/env/activate` with the correct paths for whichever tt-mlir scenario was detected.

### 2. Activate Environment

```bash
source build/env/activate
```

**What this does:**
- Sets `TT_LANG_HOME` to tt-lang project root
- Sets `TTMLIR_TOOLCHAIN_DIR` to the detected or configured toolchain directory
- Activates the Python virtual environment from the toolchain
- Sets `TTLANG_ENV_ACTIVATED=1`
- Prepends `tt-lang/build/bin` to PATH
- Prepends `tt-lang/build/python_packages` and `tt-lang/python` to PYTHONPATH
- Shows which tt-mlir is being used (build tree, installed, or locally built)

### 3. Build tt-lang

```bash
cmake --build build
```

**Configuration Options:**

- `CMAKE_BUILD_TYPE` (default: Release) - Build type (Debug, Release, RelWithDebInfo, Asan, Coverage, Assert)
- `TTMLIR_BUILD_DIR` - Path to tt-mlir build directory (Scenario 1)
- `TTMLIR_TOOLCHAIN_DIR` (default: `/opt/ttmlir-toolchain`) - Location of tt-mlir toolchain (Scenarios 2 & 3)
  - Can be set as environment variable: `export TTMLIR_TOOLCHAIN_DIR=/path/to/toolchain`
  - Or as CMake variable: `-DTTMLIR_TOOLCHAIN_DIR=/path/to/toolchain`
  - Environment variable takes precedence
  - **Recommended for Scenario 1**: Set as environment variable to prevent incorrect derivation from Python path
- `TTMLIR_INSTALL_PREFIX` (default: `${CMAKE_BINARY_DIR}/tt-mlir-install`) - Installation prefix for automatically built tt-mlir (Scenario 3 only)
- `TTMLIR_SRC_DIR` - Path to existing tt-mlir source directory (Scenario 3 only, avoids re-downloading)
- `TTMLIR_GIT_TAG` - tt-mlir commit to fetch (Scenario 3 only, overrides `third-party/tt-mlir.commit`)
- `TTLANG_ENABLE_BINDINGS_PYTHON` (default: OFF) - Enable Python bindings
- `TTLANG_ENABLE_RUNTIME` (default: OFF) - Enable runtime support
- `TTLANG_ENABLE_PERF_TRACE` (default: OFF) - Enable performance trace (Scenario 3 only, passed to tt-mlir build)
- `CODE_COVERAGE` (default: OFF) - Enable code coverage reporting

**Examples:**

```bash
# Scenario 1: Use pre-built tt-mlir (with explicit toolchain)
export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain
cmake -GNinja -Bbuild . -DTTMLIR_BUILD_DIR=/path/to/tt-mlir/build

# Scenario 2: Use pre-installed tt-mlir (via environment variable)
export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain
cmake -GNinja -Bbuild .

# Scenario 2: Use pre-installed tt-mlir (via CMake variable)
cmake -GNinja -Bbuild . -DTTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain

# Scenario 3: Automatic build (no extra options needed)
cmake -GNinja -Bbuild .

# Scenario 3: Automatic build with custom install prefix
cmake -GNinja -Bbuild . -DTTMLIR_INSTALL_PREFIX=/tmp/my-ttmlir-install

# Scenario 3: Automatic build with performance trace enabled
cmake -GNinja -Bbuild . -DTTLANG_ENABLE_PERF_TRACE=ON -DTTMLIR_CMAKE_BUILD_TYPE=Release

# Debug build with Python bindings
cmake -GNinja -Bbuild . -DCMAKE_BUILD_TYPE=Debug -DTTLANG_ENABLE_BINDINGS_PYTHON=ON
```

This builds:
- C++ libraries (when added to `lib/`)
- Python bindings (in `build/python_packages/ttlang/`)

### 4. Use Python Packages

After building, Python packages are available via PYTHONPATH:

```bash
# Already set by env/activate
python3 -c "import ttl; print(ttl.__version__)"
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

### Input Variables (can be set before CMake configure):
- `TTMLIR_TOOLCHAIN_DIR` - Toolchain installation directory (e.g., `/opt/ttmlir-toolchain`)
  - Can be set as an environment variable or CMake variable (`-DTTMLIR_TOOLCHAIN_DIR=...`)
  - Environment variable takes precedence if set
  - Defaults to `/opt/ttmlir-toolchain` if not specified
  - **Important for Scenario 1**: When using a tt-mlir build tree (no install), etting this as an environment variable prevents CMake from incorrectly deriving it from the Python executable path

### Set by tt-lang's generated activate script:
- `TT_LANG_HOME` - tt-lang project root
- `TTMLIR_TOOLCHAIN_DIR` - Toolchain directory (exported for reference)
- `TTLANG_ENV_ACTIVATED` - Set to 1 when tt-lang environment is active

### Modified by tt-lang's activate script:
- `PATH` - Prepends `$TT_LANG_HOME/build/bin`
- `PYTHONPATH` - Prepends `$TT_LANG_HOME/build/python_packages` and tt-mlir Python packages
- Python virtual environment from `${TTMLIR_TOOLCHAIN_DIR}/venv` is activated

## Development Workflow

### First-time Setup

```bash
# Build tt-lang
cd /path/to/tt-lang
cmake -GNinja -Bbuild .      # Configure and generate activation script
source build/env/activate     # Activate the environment
cmake --build build           # Build tt-lang
```

### Daily Development

```bash
# In each new shell session:
cd /path/to/tt-lang
source build/env/activate

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

### Scenario 1 Issues (Pre-built tt-mlir)

#### Error: "Could not find TTMLIR in build directory"
**Solution:** Ensure tt-mlir is built; refer to tt-mlir build instructions.

Verify `TTMLIRConfig.cmake` exists at `${TTMLIR_BUILD_DIR}/lib/cmake/ttmlir/TTMLIRConfig.cmake`.

#### Warning: "TTMLIR_TOOLCHAIN_DIR differs from tt-mlir's configured installation prefix"
**Solution:** This warning indicates a mismatch between your specified `TTMLIR_TOOLCHAIN_DIR` and the one tt-mlir was configured with. The build will use tt-mlir's value. To avoid this warning:
- Set `TTMLIR_TOOLCHAIN_DIR` as an environment variable before configuring (recommended)
- Or ensure it matches tt-mlir's `CMAKE_INSTALL_PREFIX`

### Scenario 2 Issues (Pre-installed tt-mlir)

#### Error: "Could not find TTMLIR"
**Solution:** Verify tt-mlir is installed at the expected location:
```bash
ls ${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/ttmlir/TTMLIRConfig.cmake
```

If not found:
1. Install tt-mlir to the toolchain directory
2. Or specify the correct location: `-DTTMLIR_TOOLCHAIN_DIR=/path/to/installation`
3. Or let the build system fetch and build it automatically (Scenario 3)

#### Error: "Python 3 executable not found in toolchain venv"
**Solution:** Ensure the toolchain has a Python virtual environment:
```bash
ls ${TTMLIR_TOOLCHAIN_DIR}/venv/bin/python3
```

The toolchain must include a Python 3.11+ virtual environment with required packages.

### Scenario 3 Issues (Automatic Build)

#### Error: "Failed to clone tt-mlir"
**Solution:** Ensure you have:
1. Network access to GitHub
2. Git installed and configured
3. Valid commit hash in `third-party/tt-mlir.commit`

#### Error: "tt-mlir environment not activated"
**Solution:** The automatic build requires:
- An existing LLVM/MLIR toolchain at `${TTMLIR_TOOLCHAIN_DIR}`
- Python 3.11+ in `${TTMLIR_TOOLCHAIN_DIR}/venv`

Ensure these prerequisites are met before attempting automatic build.

#### Build takes too long
**Solution:** The first automatic build fetches and compiles tt-mlir, which can take 60-90 minutes. To speed up:
- Ensure ccache is installed (automatically detected and used)
- Use a pre-installed tt-mlir (Scenario 2) for faster builds
- Subsequent builds reuse the cached tt-mlir installation
- In CI, the toolchain is cached - only the first build for a new tt-mlir commit is slow

### Common Issues

#### Python import errors
**Solution:** Ensure Python can find the packages:
```bash
# Check Python executable
which python3

# Verify it's using the toolchain venv
python3 -c "import sys; print(sys.prefix)"  # Should show TTMLIR_TOOLCHAIN_DIR/venv

# Test imports
python3 -c "import ttmlir; import ttl"
```

If imports fail, verify the build completed successfully and Python packages were installed.

#### Build errors about missing LLVM/MLIR
**Solution:** Ensure the toolchain directory contains LLVM/MLIR:
```bash
ls ${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/mlir/MLIRConfig.cmake
ls ${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/llvm/LLVMConfig.cmake
```

If missing, you need to install or build the LLVM/MLIR toolchain first.

#### CMake configuration errors
**Solution:** Clean the build directory and reconfigure:
```bash
rm -rf build
cmake -GNinja -Bbuild .
```

For persistent issues, check:
1. CMake version is 3.24 or newer
2. Ninja is installed
3. Clang/Clang++ are available
4. All paths in error messages are valid
