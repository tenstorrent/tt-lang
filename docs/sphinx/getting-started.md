# Getting Started

The fastest way to get started with TT-Lang is to install a pre-built package
from PyPI. To develop TT-Lang itself or debug the compiler, use the Docker
images or [build from source](#building-without-docker).

## Install from PyPI

We provide two TT-Lang packages: the
[tt-lang](https://pypi.org/project/tt-lang/) package includes the TT-Lang
compiler and Tenstorrent hardware support and depends on `ttnn`, `pytorch`, and
several smaller Python packages, while
[tt-lang-sim](https://pypi.org/project/tt-lang-sim/) includes only the
functional simulator (no compiler or hardware support) and does not depend on
`ttnn`.

First, create an isolated Python environment (venv, conda, etc.) with Python
3.11 or later (Python 3.12 recommended). The wheel targets a specific CPython
ABI, so the venv's Python must match — invoke `python3.12` (or `python3.11`)
explicitly rather than the system default `python3`:

```bash
python3.12 -m venv --prompt ttlang ttlang-venv
source ttlang-venv/bin/activate
```

On Linux machines with Tenstorrent hardware (Linux x86_64 / aarch64):

```bash
pip install tt-lang
tt-lang-setup                     # install matching sfpi runtime + copy tutorials
```

Functional simulator only on Linux or macOS, does not require Tenstorrent
hardware:

```bash
pip install tt-lang-sim
tt-lang-setup                     # copy bundled tutorials to ./tutorials/
```

`tt-lang-setup` is idempotent (safe to run multiple times). Inside the venv it:

- Downloads the sfpi compiler that pairs with the installed `ttnn` and extracts
  it under `<venv>/.../ttnn/runtime/sfpi/` (only for the `tt-lang` package).
- Copies bundled tutorials (`elementwise`, `matmul`, `broadcast`) to
  `./tutorials/`.

For finer control, `tt-lang-setup-sfpi` runs only the sfpi step and
`tt-lang-setup-tutorials -t <DIR>` only the tutorials copy.

### Internal S3 wheels

More frequently updated development versions of `tt-lang` are available from
Tenstorrent's S3 PyPI index.

Set `TTLANG_VERSION` to a published version from the workflow summary or the
S3 package index. A version selector is required because public PyPI also hosts
`tt-lang`, and pip resolves candidates across all configured indexes. Available
versions are listed at https://pypi.eng.aws.tenstorrent.com/.

The default internal `tt-lang` wheel bundles the `ttnn` artifacts from the
toolchain used to build the wheel, so `pip install` does not pull `ttnn` from
PyPI. As with the public wheel, `tt-lang-setup` then installs the matching sfpi
runtime and copies the tutorials:

```bash
TTLANG_VERSION=<published-internal-version>
pip install \
  --extra-index-url https://pypi.eng.aws.tenstorrent.com/ \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "tt-lang==$TTLANG_VERSION"
tt-lang-setup    # downloads sfpi into the bundled ttnn tree + copies tutorials
```

Use `tt-lang-light` only when the environment already has a newer local
tt-metal source or install layout that should provide `ttnn`. The package is a
metapackage: `tt-lang-light==X` depends on the matching no-ttnn core wheel
`tt-lang==X+light`. Install either `tt-lang` or `tt-lang-light` in an
environment, not both.

```bash
TTLANG_VERSION=<published-internal-version>
pip install \
  --extra-index-url https://pypi.eng.aws.tenstorrent.com/ \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "tt-lang-light==$TTLANG_VERSION"
tt-lang-setup    # copies tutorials only; sfpi is provided by the external tt-metal
```

The `tt-lang-setup` command above copies the tutorials into `./tutorials`.
Configure a native tt-metal source/build layout before running those local
hardware examples. The `--check` option imports `ttnn` from the selected tree,
so use it only with trusted tt-metal builds:

```bash
external_tt_metal_env="$(
  tt-lang-setup-external-tt-metal \
    --tt-metal-dir /path/to/tt-metal \
    --build-dir /path/to/tt-metal/build \
    --check
)" && eval "$external_tt_metal_env"
```

Configure an install-layout tt-metal prefix similarly:

```bash
external_tt_metal_env="$(
  tt-lang-setup-external-tt-metal \
    --tt-metal-dir /path/to/tt-metal-install \
    --check
)" && eval "$external_tt_metal_env"
python tutorials/elementwise/step_4_multinode_grid_full.py
python tutorials/matmul/step_3_multinode.py
```

Validate that Python resolves both packages from the intended environment:

```bash
python -c 'import ttnn, ttl; print(ttnn.__file__, ttl.__version__)'
```

Run a tutorial example:

```bash
tt-lang-sim tutorials/elementwise/step_4_multinode_grid_full.py    # simulator (no compilation, runs on CPU)
python tutorials/elementwise/step_4_multinode_grid_full.py        # compiles and runs on hardware
```

## Build from source for the simulator only

To run the simulator from a source checkout without installing the PyPI
package:

```bash
git clone https://github.com/tenstorrent/tt-lang.git
cd tt-lang
cmake -G Ninja -B build -DTTLANG_SIM_ONLY=ON
cmake --build build
source build/env/activate
tt-lang-sim examples/eltwise_add.py
```

## Docker quick start

Two images are available:

| Image | Purpose | Can run TT-Lang programs? | Can build TT-Lang? |
|-------|---------|:-------------------------:|:-------------------:|
| **dist** | Run TT-Lang programs | Yes | No |
| **ird** | Develop and build from source | Yes | Yes |

### Running programs (dist image)

The **dist** image contains a fully built TT-Lang installation at
`/opt/ttlang-toolchain`. Use it to compile and run TT-Lang programs without
building anything.

```bash
docker run -d --name $USER-dist \
  --device=/dev/tenstorrent/0:/dev/tenstorrent/0 \
  -v /dev/hugepages:/dev/hugepages \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v $HOME:$HOME \
  ghcr.io/tenstorrent/tt-lang/tt-lang-dist-ubuntu-24-04:latest \
  sleep infinity

docker exec -it $USER-dist /bin/bash
```

The environment activates automatically on login. Run an example immediately:

```bash
python /opt/ttlang-toolchain/examples/elementwise-tutorial/step_4_multinode_grid_full.py
```

### Building from source (ird image)

The **ird** image has the pre-built toolchain (LLVM, tt-metal, Python venv) but
does not include TT-Lang itself. Clone and build against the toolchain:

```bash
docker run -d --name $USER-ird \
  --device=/dev/tenstorrent/0:/dev/tenstorrent/0 \
  -v /dev/hugepages:/dev/hugepages \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v $HOME:$HOME \
  -v $SSH_AUTH_SOCK:/ssh-agent -e SSH_AUTH_SOCK=/ssh-agent \
  ghcr.io/tenstorrent/tt-lang/tt-lang-ird-ubuntu-24-04:latest \
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
python examples/elementwise-tutorial/step_4_multinode_grid_full.py
```

## Building without Docker

### Prerequisites

- CMake 3.28+, Ninja, and Clang 17+ or GCC 12+
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

TT-Lang includes a functional simulator that runs operations as pure Python without requiring Tenstorrent hardware or the full compiler stack. Use it to validate kernel logic and debug with any Python debugger:

```bash
tt-lang-sim examples/eltwise_add.py
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

- Take a [tour](tour/index.md) to get an introduction to TT-Lang features
  from single-tile to multinode operations
- Read the [programming guide](programming-guide.md) for compiler options, print
  debugging, and performance tools
- Use [Claude Code](https://claude.com/claude-code) with the built-in
  [slash commands](claude-skills.md) to translate kernels, profile, and optimize
- Explore the `examples/` directory for complete working programs
