# tt-lang

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![Build Status](https://github.com/tenstorrent/tt-lang/workflows/CI/badge.svg)

A Python-based Domain-Specific Language (DSL) for authoring high-performance custom kernels on Tenstorrent hardware. This project is under active development — see the [functionality matrix](docs/sphinx/specs/TTLangSpecification.md#appendix-d-functionality-matrix) for current simulator and compiler support.

## 1. Vision

TT-Lang joins the Tenstorrent software ecosystem as an expressive yet ergonomic middle ground between [TT-NN](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html) and [TT-Metalium](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/index.html), aiming to provide a unified entrypoint with integrated simulation, performance analysis, and AI-assisted development tooling.

![ecosystem graph](docs/ecosystem-diagram.png)

The language is designed to support generative AI workflows and a robust tooling ecosystem: Python as the host language enables AI tools to translate GPU DSL kernels (Triton, CUDA, cuTile, TileLang) to Tenstorrent hardware more reliably than direct TT-Metalium translation, while tight integration with functional simulation will allow AI agents to propose kernel implementations, validate correctness, and iterate on configurations autonomously. Developers should be able to catch errors and performance issues in their IDE rather than on hardware, with a functional simulator to surface bugs early. Line-by-line performance metrics and data flow graphs can guide both programmers and AI agents to easily spot bottle necks and optimization opportunities.

Tenstorrent developers today face a choice between TT-NN which provides high-level operations that are straightforward to use but lack the expressivity needed for custom kernels and TT-Metalium which provides full hardware control through explicit low-level management of memory and compute. This is not a shortcoming of TT-Metalium; it is designed to be low-level and expressive, providing direct access to hardware primitives without abstraction overhead, and it serves its purpose well for developers who need that level of control. The problem is that there is no middle ground where the compiler handles what it does best—resource management, validation, optimization—while maintaining high expressivity for application-level concerns.

TT-Lang bridges this gap through progressive disclosure: simple kernels require minimal specification where the compiler infers compute API operations, NOC addressing, DST register allocation and more from high-level abstractions, while complex kernels allow developers to open the hood and craft pipelining and synchronization details directly. The primary use case is kernel fusion for model deployment. Engineers porting models through TT-NN quickly encounter operations that need to be fused for performance or patterns that TT-NN cannot express, and today this requires rewriting in TT-Metalium which takes weeks and demands undivided attention and hardware debugging expertise. TT-Lang makes this transition fast and correct: a developer can take a sequence of TT-NN operations, express the fused equivalent with explicit control over intermediate results and memory layout, validate correctness through simulation, and integrate the result as a drop-in replacement in their TT-NN graph.

## 2. Quick Start

The fastest way to try tt-lang is with the [functional simulator](docs/sphinx/simulator.md), which runs kernels as pure Python — no hardware, no compiler build required:

```bash
git clone https://github.com/tenstorrent/tt-lang.git
cd tt-lang
cmake -G Ninja -B build -DTTLANG_SIM_ONLY=ON
source build/env/activate
./bin/ttlang-sim examples/eltwise_add.py
```

To compile and run kernels on Tenstorrent hardware, use a pre-built Docker image. Two images are available:

| Image | Purpose | Can run tt-lang programs? | Can clone/build tt-lang? |
|-------|---------|:-------------------------:|:------------------------:|
| ![dist](https://img.shields.io/badge/dist-tt--lang--dist--ubuntu--22--04-brightgreen) | Run tt-lang programs | Yes | No |
| ![ird](https://img.shields.io/badge/ird-tt--lang--ird--ubuntu--22--04-blueviolet) | Develop and build tt-lang from source | Yes | Yes |

Both images can be used with `ird reserve` (see [container build docs](.github/containers/README.md) for details).

### 2.1 ![dist](https://img.shields.io/badge/dist-brightgreen) Pre-built tt-lang (for users)

Image: ghcr.io/tenstorrent/tt-lang/tt-lang-dist-ubuntu-22-04:latest ([all versions](https://github.com/tenstorrent/tt-lang/pkgs/container/tt-lang-dist-ubuntu-22-04))

The **dist** image contains a single, fully built tt-lang installation in `/opt/ttlang-toolchain`. Use it to compile and run any tt-lang program without building any of the prerequisites.

> ⚠️ **Important**: Do not attempt to build tt-lang inside a dist container — it has no build toolchain. To clone and build tt-lang yourself, use the [**ird** image](#22--development-image-for-building-tt-lang) instead.

Create the container (one-time):
```bash
docker run -d --name $USER-dist \
  --device=/dev/tenstorrent/0:/dev/tenstorrent/0 \
  -v /dev/hugepages:/dev/hugepages \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v $HOME:$HOME \
  ghcr.io/tenstorrent/tt-lang/tt-lang-dist-ubuntu-22-04:latest \
  sleep infinity
```

Open a shell:
```bash
docker exec -it $USER-dist /bin/bash
```

The environment activates automatically on login. Run an example immediately:
```bash
python /opt/ttlang-toolchain/examples/tutorial/multicore_grid_auto.py
```

To learn more, work through the [tutorial](docs/sphinx/ttl-tutorial/index.md), explore the [programming guide](docs/sphinx/programming-guide.md) for compiler options, debugging, and performance tools, or use [Claude Code](https://claude.com/claude-code) with the built-in [slash commands](docs/sphinx/claude-skills.md) to translate kernels, profile, and optimize.

### 2.2 ![ird](https://img.shields.io/badge/ird-blueviolet) Development image (for building tt-lang)

Image: ghcr.io/tenstorrent/tt-lang/tt-lang-ird-ubuntu-22-04:latest ([all versions](https://github.com/tenstorrent/tt-lang/pkgs/container/tt-lang-ird-ubuntu-22-04))

The **ird** image has the pre-built toolchain (LLVM, tt-metal, Python venv) but does not include tt-lang itself. Clone the repository and build against the toolchain. You can maintain multiple clones or branches side by side, each with its own build directory.

To use directly with docker on your local linux machine, first create a container (one-time):
```bash
docker run -d --name $USER-ird \
  --device=/dev/tenstorrent/0:/dev/tenstorrent/0 \
  -v /dev/hugepages:/dev/hugepages \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v $HOME:$HOME \
  -v $SSH_AUTH_SOCK:/ssh-agent -e SSH_AUTH_SOCK=/ssh-agent \
  ghcr.io/tenstorrent/tt-lang/tt-lang-ird-ubuntu-22-04:latest \
  sleep infinity
```

Open a shell:
```bash
docker exec -it $USER-ird /bin/bash
```

Inside the container, clone and build:
```bash
git clone https://github.com/tenstorrent/tt-lang.git
cd tt-lang
cmake -G Ninja -B build -DTTLANG_USE_TOOLCHAIN=ON
source build/env/activate
cmake --build build
```

Verify the build:
```bash
ninja -C build check-ttlang-all
```

Run an example:
```bash
python examples/tutorial/multicore_grid_auto.py
```

The `-DTTLANG_USE_TOOLCHAIN=ON` flag tells CMake to use the pre-built LLVM and tt-metal from `/opt/ttlang-toolchain` instead of building them from source, which saves significant build time.

Performance tracing (Tracy) is enabled by default. To disable it, add `-DTTLANG_ENABLE_PERF_TRACE=OFF` to the cmake configure command. See the [programming guide](docs/sphinx/programming-guide.md) for profiling usage.

### 2.3 Building without Docker

To build tt-lang directly on a host machine without Docker, see the [build system documentation](docs/sphinx/build.md). It covers prerequisites, all supported build modes (from submodules, reusable toolchain, pre-built toolchain), and version compatibility.

### 2.4 Container Tips

To map a different TT device, change the `--device` argument (e.g., `--device=/dev/tenstorrent/1:/dev/tenstorrent/0`).

### 2.5 Functional Simulator

tt-lang includes a functional simulator that runs kernels as pure Python, without requiring Tenstorrent hardware or the full compiler stack. Use it to validate kernel logic and debug with any Python debugger:

```bash
./bin/ttlang-sim examples/eltwise_add.py
```

The simulator typically supports more language features than the compiler at any given point — see the [functionality matrix](docs/sphinx/specs/TTLangSpecification.md#appendix-d-functionality-matrix) for current coverage. See the [programming guide](docs/sphinx/simulator.md) for debugger setup and details.

## 3. Documentation

Full documentation is built with Sphinx. The source lives in [docs/sphinx/](docs/sphinx/) and covers:

- [Tutorial](docs/sphinx/ttl-tutorial/index.md) — step-by-step examples from single-tile to multinode kernels
- [Programming Guide](docs/sphinx/programming-guide.md) — compiler options, print debugging, performance tools
- [Functional Simulator](docs/sphinx/simulator.md) — run kernels without hardware, debugging setup
- [Claude Skills](docs/sphinx/claude-skills.md) — AI-assisted kernel translation, profiling, and optimization via [Claude Code](https://claude.com/claude-code)
- [Build System](docs/sphinx/build.md) — build configuration, toolchain modes, and version compatibility
- [Testing](docs/sphinx/testing.md) — how to write and run tests
- [Contributor Guide](docs/sphinx/contributor-guide.md) — workflow, validation, adding new ops

To build and view the Sphinx docs locally:
```bash
cmake -G Ninja -B build -DTTLANG_ENABLE_DOCS=ON
cmake --build build --target ttlang-docs
python -m http.server 8000 -d build/docs/sphinx/_build/html
```

## 4. Contributing

We welcome contributions. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### 4.1 Developer Guidelines

See the Sphinx [contributor guide](docs/sphinx/contributor-guide.md) and [code style guidelines](docs/sphinx/guidelines.md) for coding standards, dialect design patterns, and testing practices.

### 4.2 Updating Submodule Versions

tt-mlir defines the compatible versions of LLVM and tt-metal. When updating tt-mlir, the other submodules should be updated to match.

Update tt-mlir (and read the versions it expects):
```bash
cd third-party/tt-mlir && git fetch && git checkout <commit> && cd ../..

# Read the LLVM and tt-metal commits that this tt-mlir version expects:
grep LLVM_PROJECT_VERSION third-party/tt-mlir/env/CMakeLists.txt
grep TT_METAL_VERSION third-party/tt-mlir/third_party/CMakeLists.txt
```

Update LLVM to the compatible version:
```bash
cd third-party/llvm-project && git fetch && git checkout <llvm-sha> && cd ../..
```

Update tt-metal to the compatible version:
```bash
cd third-party/tt-metal && git fetch && git checkout <tt-metal-sha> && cd ../..
```

Commit all submodule updates together:
```bash
git add third-party/tt-mlir third-party/llvm-project third-party/tt-metal
git commit -m "Update submodules to tt-mlir <commit>"
```

The build system verifies SHA compatibility during configure. If submodule versions are intentionally mismatched, pass `-DTTLANG_ACCEPT_LLVM_MISMATCH=ON` or `-DTTLANG_ACCEPT_TTMETAL_MISMATCH=ON` to suppress the check.

### 4.3 Code Formatting with Pre-commit

tt-lang uses [pre-commit](https://pre-commit.com/) to format code and enforce style guidelines before commits.

Install and activate:
```bash
pip install pre-commit
cd /path/to/tt-lang
pre-commit install
```

Pre-commit runs automatically on `git commit`. It formats Python code with [Black](https://github.com/psf/black), C++ code with [clang-format](https://clang.llvm.org/docs/ClangFormat.html) (LLVM style), removes trailing whitespace, and checks YAML/TOML syntax.

If pre-commit modifies files, the commit is stopped. Stage the changes and commit again:
```bash
git add -u
git commit -m "Your commit message"
```

To run manually on all files: `pre-commit run --all-files`

### 4.4 Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code and treat all community members with respect.

## 5. Support

- [GitHub Issues](https://github.com/tenstorrent/tt-lang/issues) — report bugs or request features

## 6. License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

Third-party dependencies and their licenses are listed in the [NOTICE](NOTICE) file.
