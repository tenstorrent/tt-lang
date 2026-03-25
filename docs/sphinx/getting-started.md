# Getting Started

The fastest way to get started with tt-lang is with the [functional simulator](simulator.md), which runs kernels as pure Python — no Tenstorrent hardware, no compiler build required:

```bash
git clone https://github.com/tenstorrent/tt-lang.git
cd tt-lang
cmake -G Ninja -B build -DTTLANG_SIM_ONLY=ON
source build/env/activate
./bin/ttlang-sim examples/eltwise_add.py
```

To compile and run kernels on hardware, use a pre-built Docker image or build from source as described below.

## Docker quick start

Two images are available:

| Image | Purpose | Can run tt-lang programs? | Can build tt-lang? |
|-------|---------|:-------------------------:|:-------------------:|
| **dist** | Run tt-lang programs | Yes | No |
| **ird** | Develop and build from source | Yes | Yes |

### Running programs (dist image)

The **dist** image contains a fully built tt-lang installation at
`/opt/ttlang-toolchain`. Use it to compile and run tt-lang programs without
building anything.

```bash
docker run -d --name $USER-dist \
  --device=/dev/tenstorrent/0:/dev/tenstorrent/0 \
  -v /dev/hugepages:/dev/hugepages \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v $HOME:$HOME \
  ghcr.io/tenstorrent/tt-lang/tt-lang-dist-ubuntu-22-04:latest \
  sleep infinity

docker exec -it $USER-dist /bin/bash
```

The environment activates automatically on login. Run an example immediately:

```bash
python /opt/ttlang-toolchain/examples/tutorial/multicore_grid_auto.py
```

### Building from source (ird image)

The **ird** image has the pre-built toolchain (LLVM, tt-metal, Python venv) but
does not include tt-lang itself. Clone and build against the toolchain:

```bash
docker run -d --name $USER-ird \
  --device=/dev/tenstorrent/0:/dev/tenstorrent/0 \
  -v /dev/hugepages:/dev/hugepages \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v $HOME:$HOME \
  -v $SSH_AUTH_SOCK:/ssh-agent -e SSH_AUTH_SOCK=/ssh-agent \
  ghcr.io/tenstorrent/tt-lang/tt-lang-ird-ubuntu-22-04:latest \
  sleep infinity

docker exec -it $USER-ird /bin/bash
```

Inside the container:

```bash
git clone https://github.com/tenstorrent/tt-lang.git
cd tt-lang
cmake -G Ninja -B build -DTTLANG_USE_TOOLCHAIN=ON
source build/env/activate
cmake --build build
```

Verify the build and run an example:

```bash
ninja -C build check-ttlang-all
python examples/tutorial/multicore_grid_auto.py
```

## Building without Docker

### Prerequisites

- CMake 3.28+, Ninja, and Clang 17+ or GCC 11+
- Python 3.11+
- For faster builds: a pre-built toolchain at `TTLANG_TOOLCHAIN_DIR` (default
  `/opt/ttlang-toolchain`). Without one, LLVM and tt-metal build from submodules
  on first configure.

### With pre-built toolchain

```bash
cmake -G Ninja -B build -DTTLANG_USE_TOOLCHAIN=ON
source build/env/activate
cmake --build build
```

### From submodules

```bash
cmake -G Ninja -B build
source build/env/activate
cmake --build build
```

See the [build system documentation](build.md) for all supported build modes and
CMake options.

## Functional simulator

tt-lang includes a functional simulator that runs kernels as pure Python without requiring Tenstorrent hardware or the full compiler stack. Use it to validate kernel logic and debug with any Python debugger:

```bash
./bin/ttlang-sim examples/eltwise_add.py
python -m pytest test/sim/
```

The simulator typically supports more language features than the compiler at any given point — see the [functionality matrix](specs/TTLangSpecification.md#appendix-d-functionality-matrix) for current coverage. See the [programming guide](simulator.md) for debugger setup and more details.

## Quick checks

- Full compiler suite: `ninja -C build check-ttlang-all`
- MLIR tests only: `ninja -C build check-ttlang-mlir`
- Single MLIR test: `llvm-lit test/ttlang/Dialect/TTL/IR/ops.mlir`
- Simulator tests: `python -m pytest test/sim -q` (not included in
  `check-ttlang-all`)

## Next steps

- Work through the [tutorial](ttl-tutorial/index.md) for step-by-step examples
  from single-tile to multinode kernels
- Read the [programming guide](programming-guide.md) for compiler options, print
  debugging, and performance tools
- Use [Claude Code](https://claude.com/claude-code) with the built-in
  [slash commands](claude-skills.md) to translate kernels, profile, and optimize
- Explore the `examples/` directory for complete working programs
