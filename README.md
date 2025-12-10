# tt-lang (⚠️ in early development ⚠️)

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![Build Status](https://github.com/tenstorrent/tt-lang/workflows/CI/badge.svg)

A Python-based Domain-Specific Language (DSL) for authoring high-performance custom kernels on Tenstorrent hardware. **This project is currently in early development stages, the language spec has not yet been finalized and programs are not yet expected to run.**

## Vision

TT-Lang joins the Tenstorrent software ecosystem as an expressive yet ergonomic middle ground between TTNN and Metalium, aiming to provide a unified entrypoint with integrated simulation, performance analysis, and AI-assisted development tooling.

![ecosystem graph](docs/ecosystem-diagram.png)

The language is designed to support generative AI workflows and a robust tooling ecosystem: Python as the host language enables AI tools to translate GPU DSL kernels (Triton, CUDA) to Tenstorrent hardware more reliably than direct Metalium translation, while tight integration with functional simulation will allow AI agents to propose kernel implementations, validate correctness, and iterate on configurations autonomously. Developers should be able to catch errors and performance issues in their IDE rather than on hardware, with a functional simulator to surface bugs early. Line-by-line performance metrics and data flow graphs can guide both programmers and AI agents to easily spot bottle necks and optimization opportunities.

Tenstorrent developers today face a choice between TTNN which provides high-level operations that are straightforward to use but lack the expressivity needed for custom kernels and Metalium which provides full hardware control through explicit low-level management of memory and compute. This is not a shortcoming of Metalium; it is designed to be low-level and expressive, providing direct access to hardware primitives without abstraction overhead, and it serves its purpose well for developers who need that level of control. The problem is that there is no middle ground where the compiler handles what it does best—resource management, validation, optimization—while maintaining high expressivity for application-level concerns.

TT-Lang bridges this gap through progressive disclosure: simple kernels require minimal specification where the compiler infers circular buffer lifetimes, grid configurations, and resource allocation from high-level intent, while complex kernels allow developers to open the hood and specify pipelining, memory layout, and synchronization details directly. The primary use case is kernel fusion for model deployment. Engineers porting models through TTNN quickly encounter operations that need to be fused for performance or patterns that TTNN cannot express, and today this requires rewriting in Metalium which takes weeks and demands hardware expertise. TT-Lang makes this transition fast and correct: a developer can take a sequence of TTNN operations, express the fused equivalent with explicit control over intermediate results and memory layout, validate correctness through simulation, and integrate the result as a drop-in replacement in their TTNN graph.

## Features

- **Python-native DSL** - Write kernels in Python with the `@pykernel_gen` decorator
- **Tile-level operations** - Direct control over data movement and compute at the tile level
- **Hardware abstraction** - D2M (Data-to-Matmul) dialect for explicit DMA and compute operations
- **MLIR-based compilation** - Leverages LLVM/MLIR infrastructure for optimization and code generation
- **Flexible memory management** - Control over L1, DRAM, and circular buffer allocation
- **Multi-core synchronization** - Semaphores and barriers for coordinating across cores

## Prerequisites

* [CMake](https://cmake.org/) 3.28+
* [Clang](https://clang.llvm.org/) 18+ or [GCC](https://gcc.gnu.org/) 11+
* An existing LLVM/MLIR toolchain at `TTMLIR_TOOLCHAIN_DIR` (default: `/opt/ttmlir-toolchain`)
* [Python](https://www.python.org/) 3.11+ in the toolchain's virtual environment
* Optional (recommended): [Ninja](https://ninja-build.org/) build system

## Quick Start

tt-lang depends on [tt-mlir](https://github.com/tenstorrent/tt-mlir), the MLIR-based compiler infrastructure for Tenstorrent hardware. tt-mlir provides the core MLIR dialects, compilation passes, and runtime support that tt-lang builds upon to deliver a Python-based DSL for authoring custom kernels.

The build system supports three different integration scenarios for tt-mlir -- build-based, installation-based, or automatically fetched and installed (for more details on these, please refer to the [build system document](docs/BUILD_SYSTEM.md)).

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

## Example

Here's a simple example of using tt-lang to write a custom kernel:

```python
from ttlang import pykernel_gen, TensorBlock

@pykernel_gen
def my_kernel(input_tensor: TensorBlock, output_tensor: TensorBlock):
    # Custom kernel implementation with tile-level operations
    pass
```

See the `examples/` directory for complete working examples, including:
- `custom_dm_matmul.py` - Custom data movement with matrix multiplication
- Additional examples demonstrating DMA operations, circular buffers, and multi-core patterns

## Documentation

- [Hitchhiker's Guide](docs/HITCHHIKERS_GUIDE.md) - Complete DSL guide with examples and pipeline architecture
- [Build System](docs/BUILD_SYSTEM.md) - Detailed build configuration options and integration scenarios
- [Testing Guide](test/TESTING.md) - How to write and run tests using LLVM lit

## Testing

Run the test suite using LLVM's lit framework:

```bash
source build/env/activate
llvm-lit -sv test/python/
```

For more information on testing, including how to write new tests and interpret results, see [test/TESTING.md](test/TESTING.md).

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

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to tt-lang.

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code and treat all community members with respect.

## Support

- **Issues:** [GitHub Issues](https://github.com/tenstorrent/tt-lang/issues) - Report bugs or request features

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Third-party dependencies and their licenses are listed in the [NOTICE](NOTICE) file.
