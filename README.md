# tt-lang
TT-Lang is a Python-based DSL that enables authoring of programs for TT hardware.

See [RFC document](https://docs.google.com/document/d/1T8Htv9nfdufebajJidzYHfDfsSnru_9xn_whdyxn6Po/edit?usp=sharing).

## Build System

tt-lang uses a CMake-based build system that reuses tt-mlir's environment and toolchain. See [BUILD_SYSTEM.md](BUILD_SYSTEM.md) for complete documentation.

## Quick Start

### Prerequisites

**tt-mlir must be built first.** tt-lang depends on tt-mlir and reuses its toolchain. Please refer to the [Getting Started Guide](https://docs.tenstorrent.com/tt-mlir/getting-started.html) on how to build tt-mlir and its prerequisites. Note that the brief instructions below do not cover all possible build scenarios.

1. Clone the correct version of tt-mlir; make sure to use the version in [third-party/tt-mlir.
commit](./third-party/tt-mlir.commit) (different versions are not guaranteed to be compatible).

```bash

git clone https://github.com/tenstorrent/tt-mlir.git
cd tt-mlir
git checkout <commit sha from third-party/tt-mlir.commit>
```

# Activate tt-mlir environment and build tt-mlir
source env/activate
cmake -GNinja -Bbuild .
cmake --build build
```

2. Configure and build tt-lang.

```bash
cd /path/to/tt-lang
source env/activate
cmake -GNinja -Bbuild .
cmake --build build
```

**Build options:**
```bash
cmake -GNinja -Bbuild . -DCMAKE_BUILD_TYPE=Debug -DTTLANG_ENABLE_BINDINGS_PYTHON=ON
```

**Note:** The `third-party/tt-mlir.commit` file contains a reference tt-mlir version for compatibility. Ensure your installed tt-mlir is compatible.

## Python Package Structure

The `ttlang` Python package provides a DSL for authoring custom data movement and compute kernels:

```
python/ttlang/
├── __init__.py           # Main package exports
├── d2m_api.py            # Core decorator and compilation orchestration
├── operators.py          # TensorBlock, CircularBuffer, MemTx, Semaphore, DMA operations
├── layouts.py            # MetalLayoutAttr creation and stream layout utilities
├── codegen.py            # D2M generic function creation and code generation
├── dtype_utils.py        # PyTorch/runtime data type conversions
├── templates.py          # Common kernel templates (matmul, elementwise)
├── constants.py          # Shared constants (tile sizes, memory spaces)
└── _src/                 # Internal implementation modules
    ├── d2m_ast.py        # D2M dialect AST compiler
    ├── kernel_ast.py     # Base kernel compilation infrastructure
    ├── kernel_types.py   # CircularBuffer, Kernel, and other types
    ├── base_ast.py       # AST base classes
    ├── stream.py         # Stream type for async data movement
    └── utils.py          # Utility functions
```

See [docs/HITCHHIKERS_GUIDE.md](docs/HITCHHIKERS_GUIDE.md) for comprehensive DSL documentation and examples.

## Developer Guidelines

### Updating tt-mlir version

Update the `third-party/tt-mlir.commit` file to the desired SHA. Repeat steps 1-3 above after checking out that SHA in your tt-mlir clone.

In future, the tt-lang build will be extended to automatically fetch and build the supported tt-mlir version.

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

This will configure git to run `pre-commit` checks before each commit.

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
