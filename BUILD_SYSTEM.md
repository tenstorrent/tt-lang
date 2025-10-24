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

## Prerequisites

**tt-mlir must be built first!** tt-lang depends on tt-mlir and reuses its toolchain.

1. Clone and build tt-mlir:
   ```bash
   cd /Users/bnorris/tt
   git clone https://github.com/tenstorrent/tt-mlir.git
   cd tt-mlir
   
   # Build tt-mlir's toolchain (LLVM/MLIR, etc.)
   cmake -GNinja -Bbuild-env env/
   cmake --build build-env
   
   # Activate tt-mlir environment
   source env/activate
   
   # Build tt-mlir
   cmake -GNinja -Bbuild .
   cmake --build build
   ```

2. Verify tt-mlir is working:
   ```bash
   source env/activate
   ttmlir-opt --version
   ```

## Directory Structure

```
tt-lang/
├── CMakeLists.txt                 # Root build file
├── BUILD_SYSTEM.md                # This file
├── README.md                      # Project README
├── requirements.txt               # Python runtime requirements
├── dev-requirements.txt           # Development requirements
├── pytest.ini                     # Pytest configuration
├── env/
│   ├── activate                   # Environment activation (sources tt-mlir's env)
│   └── activate.fish              # Fish shell activation script
├── sim/
│   └── src/
│       └── cbsim/                 # Simulator code (Python)
├── include/
│   ├── CMakeLists.txt
│   ├── ttlang/                    # Public C++ headers (for future dialects)
│   └── ttlang-c/                  # Public C API headers
├── lib/
│   └── CMakeLists.txt             # C++ libraries (for future dialects/passes)
├── python/
│   ├── CMakeLists.txt             # Python bindings build
│   ├── pyproject.toml             # Python project configuration
│   ├── setup.py                   # Python package setup
│   └── ttlang/                    # Python package
│       └── __init__.py
├── third-party/
│   └── CMakeLists.txt             # Third-party dependencies
├── test/
│   └── CMakeLists.txt             # Lit test configuration (placeholder)
└── tests/
    └── test_cbsim.py              # Existing Python tests
```

## Build Process

### 1. Activate Environment

The tt-lang environment sources tt-mlir's environment and adds tt-lang-specific paths:

```bash
cd /Users/bnorris/tt/tt-lang
source env/activate
```

**What this does:**
- Sets `TT_MLIR_HOME` (auto-detects if in `/Users/bnorris/tt/tt-mlir` or `../tt-mlir`)
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

### Using tt-mlir Python Bindings

tt-mlir's Python bindings are available in the shared Python environment:

```python
# Both tt-mlir and tt-lang bindings are available
import ttlang  # tt-lang bindings
import ttmlir  # tt-mlir bindings (from shared environment)
```

## Build Types

The build system supports tt-mlir's custom build types:

- **Debug** - Debug symbols, assertions enabled, debug logs enabled
- **Release** - Optimized, no debug symbols (default)
- **RelWithDebInfo** - Optimized with debug symbols
- **Asan** - Address Sanitizer enabled for memory debugging
- **Coverage** - Code coverage instrumentation enabled
- **Assert** - Release build with assertions

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
# 1. Ensure tt-mlir is built (see Prerequisites section)
cd /Users/bnorris/tt/tt-mlir
source env/activate
cmake -GNinja -Bbuild .
cmake --build build

# 2. Set up tt-lang
cd /Users/bnorris/tt/tt-lang
source env/activate  # This sources tt-mlir's env automatically
cmake -GNinja -Bbuild .
cmake --build build
```

### Daily Development

```bash
# In each new shell session:
cd /Users/bnorris/tt/tt-lang
source env/activate

# Build
cmake --build build

# Test
pytest tests/
```

### Rebuilding After Changes

```bash
# After CMakeLists.txt changes:
cmake -GNinja -Bbuild .

# After C++ code changes:
cmake --build build

# After Python code changes (if no C++ changes):
# Python files are used directly from source or build/python_packages
```

### Updating tt-mlir Dependency

```bash
# 1. Update tt-mlir
cd /Users/bnorris/tt/tt-mlir
git pull  # or checkout specific commit/tag
cmake --build build

# 2. Rebuild tt-lang (might need reconfigure if tt-mlir APIs changed)
cd /Users/bnorris/tt/tt-lang
source env/activate
cmake -GNinja -Bbuild .  # Reconfigure to pick up new tt-mlir
cmake --build build
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
- Add lit tests in `test/` directory
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

## Benefits of This Approach

1. **Faster setup** - No need to build LLVM/MLIR separately
2. **Disk space** - Saves ~50GB by not duplicating LLVM/MLIR builds
3. **Consistency** - Same compiler, same LLVM version, same Python packages
4. **Simplified maintenance** - One toolchain to update/maintain
5. **Easy upgrades** - Update tt-mlir, rebuild, done
6. **Shared Python environment** - Can use both tt-mlir and tt-lang bindings together
