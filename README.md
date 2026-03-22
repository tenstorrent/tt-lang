# tt-lang (⚠️ in early development ⚠️)

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![Build Status](https://github.com/tenstorrent/tt-lang/workflows/CI/badge.svg)

A Python-based Domain-Specific Language (DSL) for authoring high-performance custom kernels on Tenstorrent hardware. **This project is currently in early development stages, the language spec has not yet been finalized and programs are not yet expected to run.**

## Vision

TT-Lang joins the Tenstorrent software ecosystem as an expressive yet ergonomic middle ground between [TT-NN](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html) and [TT-Metalium](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/index.html), aiming to provide a unified entrypoint with integrated simulation, performance analysis, and AI-assisted development tooling.

![ecosystem graph](docs/ecosystem-diagram.png)

The language is designed to support generative AI workflows and a robust tooling ecosystem: Python as the host language enables AI tools to translate GPU DSL kernels (Triton, CUDA, cuTile, TileLang) to Tenstorrent hardware more reliably than direct TT-Metalium translation, while tight integration with functional simulation will allow AI agents to propose kernel implementations, validate correctness, and iterate on configurations autonomously. Developers should be able to catch errors and performance issues in their IDE rather than on hardware, with a functional simulator to surface bugs early. Line-by-line performance metrics and data flow graphs can guide both programmers and AI agents to easily spot bottle necks and optimization opportunities.

Tenstorrent developers today face a choice between TT-NN which provides high-level operations that are straightforward to use but lack the expressivity needed for custom kernels and TT-Metalium which provides full hardware control through explicit low-level management of memory and compute. This is not a shortcoming of TT-Metalium; it is designed to be low-level and expressive, providing direct access to hardware primitives without abstraction overhead, and it serves its purpose well for developers who need that level of control. The problem is that there is no middle ground where the compiler handles what it does best—resource management, validation, optimization—while maintaining high expressivity for application-level concerns.

TT-Lang bridges this gap through progressive disclosure: simple kernels require minimal specification where the compiler infers compute API operations, NOC addressing, DST register allocation and more from high-level abstractions, while complex kernels allow developers to open the hood and craft pipelining and synchronization details directly. The primary use case is kernel fusion for model deployment. Engineers porting models through TT-NN quickly encounter operations that need to be fused for performance or patterns that TT-NN cannot express, and today this requires rewriting in TT-Metalium which takes weeks and demands undivided attention and hardware debugging expertise. TT-Lang makes this transition fast and correct: a developer can take a sequence of TT-NN operations, express the fused equivalent with explicit control over intermediate results and memory layout, validate correctness through simulation, and integrate the result as a drop-in replacement in their TT-NN graph.

## Prerequisites

* [CMake](https://cmake.org/) 3.28+
* [Clang](https://clang.llvm.org/) 18+ or [GCC](https://gcc.gnu.org/) 11+
* [Python](https://www.python.org/) 3.11+
* [Git](https://git-scm.com/) (for submodule checkout)
* Optional (recommended): [Ninja](https://ninja-build.org/) build system
* Optional: Pre-built LLVM/MLIR toolchain at `TTLANG_TOOLCHAIN_DIR` (use with `-DTTLANG_USE_TOOLCHAIN=ON`)

## Quick Start

tt-lang uses LLVM/MLIR and a small subset of [tt-mlir](https://github.com/tenstorrent/tt-mlir) dialects (TTCore, TTKernel, TTMetal) plus [tt-metal](https://github.com/tenstorrent/tt-metal) for runtime support. All dependencies are built from git submodules during cmake configure.

The build system supports three modes: build from submodules (default), pre-built toolchain (`TTLANG_USE_TOOLCHAIN`), or pre-built LLVM/MLIR prefix (`MLIR_PREFIX`). See the [build system document](docs/BUILD_SYSTEM.md) for details.

```bash
cd /path/to/tt-lang
cmake -GNinja -Bbuild .
source build/env/activate
cmake --build build
```

LLVM is built and installed to `build/llvm-install/` by default. tt-metal builds to `third-party/tt-metal/build/`. tt-mlir dialects compile inline. The generated `env/activate` script in tt-lang's build directory sets up all paths automatically.

**Build options:**

To use a custom toolchain directory (e.g. `/opt/ttlang-toolchain`), create it first with your user's permissions:
```bash
sudo install -d -o $(id -u) -g $(id -g) /opt/ttlang-toolchain
```

```bash
# Debug build
cmake -GNinja -Bbuild . -DCMAKE_BUILD_TYPE=Debug

# Build toolchain (LLVM + tt-metal) into a reusable prefix
cmake -GNinja -Bbuild . -DTTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain

# Use a previously built toolchain (skips LLVM + tt-metal build)
cmake -GNinja -Bbuild . -DTTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain -DTTLANG_USE_TOOLCHAIN=ON

# Force rebuild of LLVM + tt-metal into toolchain dir (ignores cached state)
cmake -GNinja -Bbuild . -DTTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain -DTTLANG_FORCE_TOOLCHAIN_REBUILD=ON

# Use a pre-built LLVM/MLIR installation at a custom path
cmake -GNinja -Bbuild . -DMLIR_PREFIX=/path/to/llvm-install
```

To generate the Sphinx documentation, configure with `-DTTLANG_ENABLE_DOCS`.

**Version compatibility:** Dependencies are pinned as git submodules under `third-party/`. The build system verifies that submodule SHAs match the versions expected by tt-mlir. If a mismatch is detected (e.g., after updating a submodule independently), the configure step will fail with a diagnostic. To proceed despite a mismatch:
```bash
# Override LLVM SHA check
cmake -GNinja -Bbuild . -DTTLANG_ACCEPT_LLVM_MISMATCH=ON

# Override tt-metal SHA check
cmake -GNinja -Bbuild . -DTTLANG_ACCEPT_TTMETAL_MISMATCH=ON
```

## Simulator-Only Execution

For users who want to run simulator examples without building the full compiler stack:

```bash
./bin/ttlang-sim examples/eltwise_add.py
pytest test/sim/
```

## Simulator Debugging

The simulator runs as standard Python code, enabling any Python debugger to work with it. The specific setup depends on your debugger of choice.

### VSCode debugger

Create a VSCode debug configuration in `.vscode/launch.json`:

```json
{
  "name": "Debug TTL Simulator",
  "type": "debugpy",
  "request": "launch",
  "module": "sim.ttlang_sim",
  "args": ["${file}"],
  "console": "integratedTerminal",
  "justMyCode": false,
  "cwd": "${workspaceFolder}",
  "env": {
    "PYTHONPATH": "${workspaceFolder}/python"
  }
}
```

This configuration:
- Launches the simulator as a Python module (`sim.ttlang_sim`)
- Passes the currently open file as the target kernel
- Sets `justMyCode: false` to enable debugging into simulator internals
- Configures `PYTHONPATH` to locate the simulator modules

**Usage:**
1. Open a kernel file in VSCode (e.g., `examples/eltwise_add.py`)
2. Set breakpoints in your kernel code
3. Press F5 or select "Debug TTL Simulator" from the Run menu
4. The debugger stops at breakpoints, allowing variable inspection and step-through execution


## Example

See the `examples/` and `test/` directories for complete working examples, including:
- `test/python/simple_add.py`
- `test/python/simple_fused.py`

Note: this project is currently in early prototype phase, examples are not final and may change significantly as we finalize the initial language spec and implement features.

## Documentation

- [Tutorial](examples/tutorial/) - Step-by-step examples from single-tile to multinode kernels
- [Build System](docs/BUILD_SYSTEM.md) - Detailed build configuration options and integration scenarios
- [Performance Tools](docs/performance-tools.md) - Profiling, signposts, and Perfetto trace visualization
- [Testing Guide](test/TESTING.md) - How to write and run tests using LLVM lit
- [Sphinx docs](docs/README.md) - How to build, view, and extend the documentation (docs are disabled by default; enable with `-DTTLANG_ENABLE_DOCS=ON` and build with `cmake --build build --target ttlang-docs`)
- [Print Debugging](docs/print-debugging.md) - Debug kernel code with `print()` statements

## Print Debugging

Use `print()` inside kernel code to emit device debug prints. Enable at runtime with `TT_METAL_DPRINT_CORES`:

```bash
export TT_METAL_DPRINT_CORES=0,0   # core to capture
python my_kernel.py 2>&1 > output.txt
```

```python
@ttl.compute()
def compute():
    with inp_dfb.wait() as tile, out_dfb.reserve() as o:
        print("hello")                             # auto: math thread
        print(tile)                                # auto: pack thread
        result = ttl.exp(tile)
        print(_dump_dst_registers=True, label="after exp") # auto: math thread
        o.store(result)

@ttl.datamovement()
def dm_write():
    print(out_dfb)                               # CB metadata
    with out_dfb.wait() as blk:
        print(blk, num_pages=1)                  # raw tensor page
        tx = ttl.copy(blk, out[0, 0])
        tx.wait()
```

- Prints can be extremely large and slow; redirect output to a file and use grep
- Always guard compute prints with `thread=` to avoid overlapping output from the three TRISC threads
- Prints all tiles/dst reg in a block

See [docs/print-debugging.md](docs/print-debugging.md) for all supported modes (scalars, tiles, tensor pages, CB details, DST registers, thread conditioning).

## Claude Skills

> ⚠️ Skills are an experimental feature under active development; skills currently reference in-flight functionality that may not be available such as the matmul operator.

One of the easiest way to get started with tt-lang is using [Claude Code](https://claude.com/claude-code) and an existing codebase. TT-Lang provides slash commands that guide Claude through kernel translation, testing, profiling, and optimization workflows.

### Example Workflow

```bash
# Clone a model you want to port
git clone https://github.com/karpathy/nanoGPT
cd nanoGPT

# Install TT-Lang slash commands (one-time setup)
cd /path/to/tt-lang/claude-slash-commands
./install.sh

# Open Claude Code in your project
cd /path/to/nanoGPT
claude

# Now type slash to use skills to translate kernels to TT-Lang:
#   /ttl-import model.py    "translate the attention kernel to TT-Lang DSL"
```

### Available Commands

Run `/ttl-help` in Claude Code to see all available commands. Here is a summary:

```
/ttl-import <kernel>
    Translate a CUDA, Triton, or PyTorch kernel to TT-Lang DSL. Analyzes the
    source kernel, maps GPU concepts to Tenstorrent equivalents, and iterates
    on testing until the translated kernel matches the original behavior.

/ttl-export <kernel>
    Export a TT-Lang kernel to TT-Metal C++ code. Runs the compiler pipeline,
    extracts the generated C++, and beautifies it by improving variable names
    and removing unnecessary casts for readable, production-ready output.

/ttl-optimize <kernel>
    Profile a kernel and apply performance optimizations. Identifies bottlenecks,
    suggests improvements like tiling, pipelining, and fusion, then validates
    that optimizations preserve correctness while improving throughput.

/ttl-profile <kernel>
    Run the profiler and display per-line cycle counts. Shows exactly where time
    is spent in the kernel with annotated source, hotspot highlighting, and
    memory vs compute breakdown.

/ttl-bug <reproducer>
    File a bug report for TT-Lang with a reproducer.

/ttl-help
    List all available TT-Lang slash commands with descriptions and examples.
```

## Performance Tools

TT-Lang includes built-in performance analysis tools for profiling kernels on hardware:

- **Perf Summary** (`TTLANG_PERF_DUMP=1`) -- NOC traffic and per-thread wall time breakdown
- **Auto-Profiling** (`TTLANG_AUTO_PROFILE=1`) -- automatic per-line cycle count instrumentation
- **User-Defined Signposts** (`TTLANG_SIGNPOST_PROFILE=1`) -- targeted cycle counts for `ttl.signpost()` regions
- **Perfetto Trace Server** (`TTLANG_PERF_SERV=1`) -- visualize profiler data in the Perfetto UI

See [docs/performance-tools.md](docs/performance-tools.md) for usage, environment variable reference, and sample output.

## Docker Containers

Pre-built Docker images are available for running tt-lang on Tenstorrent hardware.

**Available images:**
- `ghcr.io/tenstorrent/tt-lang/tt-lang-dist-ubuntu-22-04:latest` - Pre-built tt-lang (recommended)
- `ghcr.io/tenstorrent/tt-lang/tt-lang-ird-ubuntu-22-04:latest` - Development image (build tt-lang yourself)

**Starting a container:**

Replace "dist" with "ird" for the development image. To map a different TT device, change the `--device` argument, e.g., `--device=/dev/tenstorrent/1:/dev/tenstorrent/0`.

```bash
docker run -it --name $USER-dist \
  --device=/dev/tenstorrent/0:/dev/tenstorrent/0 \
  -v /dev/hugepages:/dev/hugepages \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  ghcr.io/tenstorrent/tt-lang/tt-lang-dist-ubuntu-22-04:latest \
  /bin/bash
```

To forward your SSH agent (for git clone/push inside the container), add:
```bash
  -v $SSH_AUTH_SOCK:/ssh-agent -e SSH_AUTH_SOCK=/ssh-agent
```

**Working with a running container:**
```bash
# Open a shell
docker exec -it $USER-dist /bin/bash

# Copy files in
docker cp /path/to/files $USER-dist:/root/
```

**Using the TT IRD machine pool:**

Reserve a machine with the tt-lang container pre-loaded:
```bash
# Wormhole
ird reserve \
  --docker-image ghcr.io/tenstorrent/tt-lang/tt-lang-dist-ubuntu-22-04:latest \
  --timeout 720 wormhole_b0 --machine $(hostname) --num-pcie-chips 1 --model x2

# Blackhole
ird reserve \
  --docker-image ghcr.io/tenstorrent/tt-lang/tt-lang-dist-ubuntu-22-04:latest \
  --timeout 720 blackhole --machine $(hostname) --num-pcie-chips 1
```

## Testing

Run tests using CMake targets:

```bash
source build/env/activate

# All tests (MLIR + Python)
cmake --build build --target check-ttlang

# MLIR dialect tests only
cmake --build build --target check-ttlang-mlir

# Python runtime tests only
cmake --build build --target check-ttlang-python-lit
```

Or run specific test suites using lit directly:

```bash
source build/env/activate
llvm-lit -sv test/ttlang/     # MLIR dialect tests
llvm-lit -sv test/python/     # Python runtime tests
```

For more information on testing, including how to write new tests and interpret results, see [test/TESTING.md](test/TESTING.md).

## Compiler Options

Kernels accept compiler options that control code generation (e.g., `--no-ttl-maximize-dst`, `--no-ttl-fpu-binary-ops`). These can be passed as command-line arguments, via the `@ttl.kernel` decorator's `options=` parameter, or the `TTLANG_COMPILER_OPTIONS` environment variable. Command-line arguments take highest priority.

```bash
# List available options
python examples/tutorial/multinode_grid_auto.py --ttl-help

# Run a kernel with options
python examples/tutorial/multinode_grid_auto.py --no-ttl-maximize-dst
```

See `python/ttl/compiler_options.py` for details on priority ordering and the merge protocol.

## Developer Guidelines

### Updating submodule versions

tt-mlir defines the compatible versions of LLVM and tt-metal. When updating tt-mlir, the other submodules should be updated to match.

**Update tt-mlir** (and read the versions it expects):
```bash
cd third-party/tt-mlir && git fetch && git checkout <commit> && cd ../..

# Read the LLVM and tt-metal commits that this tt-mlir version expects:
grep LLVM_PROJECT_VERSION third-party/tt-mlir/env/CMakeLists.txt
grep TT_METAL_VERSION third-party/tt-mlir/third_party/CMakeLists.txt
```

**Update LLVM to the compatible version:**
```bash
cd third-party/llvm-project && git fetch && git checkout <llvm-sha> && cd ../..
```

**Update tt-metal to the compatible version:**
```bash
cd third-party/tt-metal && git fetch && git checkout <tt-metal-sha> && cd ../..
```

**Commit all submodule updates together:**
```bash
git add third-party/tt-mlir third-party/llvm-project third-party/tt-metal
git commit -m "Update submodules to tt-mlir <commit>"
```

The build system verifies SHA compatibility during configure. If submodule versions
are intentionally mismatched, pass `-DTTLANG_ACCEPT_LLVM_MISMATCH=ON` or
`-DTTLANG_ACCEPT_TTMETAL_MISMATCH=ON` to suppress the check.

Refer to the [BuildSystem.md](docs/BUILD_SYSTEM.md) document for details on building with a pre-built MLIR installation.

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
