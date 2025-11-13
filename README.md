# tt-lang
TT-Lang is a Python-based DSL that enables authoring of programs for TT hardware.

See [RFC document](https://docs.google.com/document/d/1T8Htv9nfdufebajJidzYHfDfsSnru_9xn_whdyxn6Po/edit?usp=sharing).

## Prerequisites

* CMake 3.28+
* Clang 18+ or gcc 11+
* An existing LLVM/MLIR toolchain at `TTMLIR_TOOLCHAIN_DIR` (default: `/opt/ttmlir-toolchain`).
* Python 3.11+ in the toolchain's virtual environment
* Optional (recommended): Ninja

## Quick Start

tt-lang depends on tt-mlir. The build system supports three different integration scenarios for tt-mlir -- build-based, installation-based, or automatically fetched and installed (for more details on these, please refer to the [build system document](BUILD_SYSTEM.md)).

Here we describe the most common scenario for tt-lang users who do not have a pre-built or pre-installed tt-mlir. Note that this will fetch, configure, build and install the tt-mlir version whose commit SHA is in `third-party/tt-mlir.commit`.

```bash
cd /path/to/tt-lang
cmake -GNinja -Bbuild .
source build/env/activate
cmake --build build
```

The tt-mlir will be built and installed to `build/tt-mlir-install/` by default (or to the location specified by `TTMLIR_INSTALL_PREFIX`). The generated `env/activate` script in tt-lang's build directory will automatically use this local installation. This process requires:
- An existing LLVM/MLIR toolchain at `TTMLIR_TOOLCHAIN_DIR` (default: `/opt/ttmlir-toolchain`)

**Build options:**
```bash
# Debug build with Python bindings
cmake -GNinja -Bbuild . -DCMAKE_BUILD_TYPE=Debug -DTTLANG_ENABLE_BINDINGS_PYTHON=ON

# Custom install prefix for automatically built tt-mlir
cmake -GNinja -Bbuild . -DTTMLIR_INSTALL_PREFIX=/tmp/my-ttmlir-install

# Enable code coverage
cmake -GNinja -Bbuild . -DCODE_COVERAGE=ON
```

**Note:** The `third-party/tt-mlir.commit` file contains the reference tt-mlir version. The build system ensures version compatibility automatically.

## Python Package Structure

The `ttlang` Python package provides a DSL for authoring custom data movement and compute kernels:

```
python/ttlang/
├── __init__.py           # Main package exports
├── d2m_api.py            # Core decorator and compilation orchestration
├── operators.py          # TensorBlock, MemTx, DMA operations
├── circular_buffer.py    # CircularBuffer for inter-thread communication
├── semaphore.py          # Semaphore for multi-core synchronization
├── layouts.py            # MetalLayoutAttr creation and accessor layout utilities
├── dtype_utils.py        # PyTorch/runtime data type conversions
├── constants.py          # Shared constants (tile sizes, memory spaces)
└── _src/                 # Internal implementation modules
    ├── d2m_ast.py        # D2M dialect AST compiler
    ├── kernel_ast.py     # Base kernel compilation infrastructure
    ├── kernel_types.py   # CircularBuffer, Kernel, and other types
    ├── base_ast.py       # AST base classes
    ├── tensor_accessor.py  # TensorAccessor type for indexed tile-level access
    ├── utils.py          # Utility functions
    └── codegen.py        # D2M generic function creation and code generation
```

See [docs/HITCHHIKERS_GUIDE.md](docs/HITCHHIKERS_GUIDE.md) for comprehensive DSL documentation and examples.

## Developer Guidelines

### Updating tt-mlir version

Update the `third-party/tt-mlir.commit` file to the desired commit SHA if using the automated tt-mlir install. Refer to the [BuildSystem.md](docs/BUILD_SYSTEM.md) document for details on building with a pre-built tt-mlir or pre-installed one.

### Code Formatting with Pre-commit

tt-lang uses [pre-commit](https://pre-commit.com/) to automatically format code and enforce style guidelines before commits.

#### Installation

Install pre-commit using pip:

```bash
pip install pre-commit
```

Or using your system package manager:
```bash
# macOS
brew install pre-commit

# Ubuntu/Debian
sudo apt install pre-commit
```

#### Setup

After cloning the repository, install the git hook scripts:

```bash
cd /path/to/tt-lang
pre-commit install
```

This will configure git to run `pre-commit` checks before each commit. You may also
choose not to do this step and instead run `pre-commit` manually as described
below.

#### Usage

Once installed, `pre-commit` will automatically run when you commit:

```bash
git commit -m "Your commit message"
```

Pre-commit will:
- Format Python code with [Black](https://github.com/psf/black)
- Format C++ code with [clang-format](https://clang.llvm.org/docs/ClangFormat.html) (LLVM style)
- Remove trailing whitespace
- Ensure files end with a single newline
- Check YAML and TOML syntax
- Check for large files
- Check for valid copyright notice

If `pre-commit` makes changes, the commit will be stopped. Review the changes, stage them, and commit again:

```bash
git add -u
git commit -m "Your commit message"
```

#### Manual Formatting

To run pre-commit checks manually on all files:

```bash
pre-commit run --all-files
```

To run on specific files:

```bash
pre-commit run --files path/to/file1.py path/to/file2.cpp
```

#### Skipping Pre-commit (Not Recommended)

In rare cases where you need to skip pre-commit checks:

```bash
git commit --no-verify -m "Your commit message"
```

**Note:** CI will still run these checks, so skipping locally may cause CI failures.
