# Getting started with compiler-backed emulation

The `emule` backend compiles TT-Lang programs and executes their generated
kernels through tt-metal and tt-emule inside Docker. It uses one supported stack:
the TT-Lang checkout, tt-emule revision, tt-metal revision, base image, and P150
target recorded in `config/tt-lang-emule-stack.json`.

Users do not select these components independently. Install the recorded
environment once, then run programs with the same `tt-lang-sim` interface used
by the Python backend.

## Host prerequisites

- Git and Python 3.10 or newer on `PATH`.
- Docker Desktop on macOS, or Docker Engine on Linux, with the daemon running
  and accessible without `sudo`.
- Support for `linux/amd64` containers. Apple Silicon uses Docker's x86-64
  virtualization; enabling Rosetta support reduces translation overhead where
  available.
- The approved tt-emule repository URL and Git access to that repository. The
  installer uses the host's existing Git credentials.

Check the host tools before installation:

```bash
git --version
python3 --version
docker info
```

## Obtain TT-Lang

The compiler-backed backend currently requires a TT-Lang source checkout. Check
out the branch or release that contains the desired stack, including its
submodules:

```bash
git clone --recurse-submodules https://github.com/tenstorrent/tt-lang.git
cd tt-lang
```

The installer validates that the checkout contains the compiler baseline
recorded in the stack manifest. It also fetches the exact
tt-emule revision and verifies that tt-emule pins the same tt-metal revision as
the manifest; it does not search for or accept another combination.

## Install the environment

The public manifest deliberately records the exact emulator commit without
publishing its source coordinate. Set the repository location supplied by the
tt-emule owners, then run the installer from the checkout root:

```bash
export TTLANG_EMULE_RUNTIME_SOURCE_URL=REPOSITORY_URL
./scripts/install-tt-lang-emule.sh
```

The URL locates the pinned source; it does not select the emulator version.
The installer checks out only the manifest's exact commit and verifies its
tt-metal pin.

Installation builds the pinned tt-emule/tt-metal Docker image and compiles this
TT-Lang checkout into a persistent Docker volume. It can take substantial time,
CPU, memory, and disk space on its first run. On success it prints the compiler
commit for which the environment was installed.

Installation is separate from execution. `tt-lang-sim` never configures or
builds TT-Lang. If the runtime image is absent, the compiler environment is
incomplete, or the checkout has moved to another commit, execution stops with
an instruction to run the installer again.

There is no per-checkout configuration file and no independently selectable
compiler, emulator, or tt-metal version. Docker's image and volume caches retain
the installed environment.

## Run a program

After installation:

```bash
./bin/tt-lang-sim --backend=emule examples/eltwise_add.py
```

The program imports the real `ttl` and `ttnn` packages from the installed
environment. The compiler generates kernels and tt-emule executes them on the
recorded emulated P150 target.

Arguments belonging to the program follow `--`:

```bash
./bin/tt-lang-sim --backend=emule program.py -- --program-option value
```

The launcher mounts the checkout and the current working directory. Relative
program inputs beneath the current working directory are portable into the
container; arbitrary absolute host paths are not rewritten.

The Python backend remains the default and does not require Docker:

```bash
./bin/tt-lang-sim program.py
./bin/tt-lang-sim --backend=python program.py
```

## Run tests with the existing test framework

`tt-lang-sim` executes programs; it does not provide a second test-selection or
reporting interface. Use TT-Lang's normal CMake, pytest, and lit commands from an
activated compiler build environment.

Run the complete compiler suite:

```bash
cmake --build build --target check-ttlang-all
```

Run the device-independent compiler suites:

```bash
cmake --build build --target check-ttlang
```

Run or select pytest tests directly, including ordinary pytest filtering and
reporting options:

```bash
pytest -v test/python
pytest -v test/me2e
pytest -v test/python/test_elementwise_ops.py -k add
```

Run lit suites or individual cases directly:

```bash
cmake --build build --target check-ttlang-mlir
llvm-lit -v build/test/python
llvm-lit -v build/test/python/simple_add.py
```

These commands use the runtime configured for that compiler build. Compiler-only
tests do not execute tt-emule; device tests require a compatible emule-enabled
Linux build environment. See
[`test/TESTING.md`](https://github.com/tenstorrent/tt-lang/blob/main/test/TESTING.md)
for the suite boundaries, device requirements, pytest selection, lit paths, and
output locations. See [Testing](testing.md) for the short command reference.

Representative programs do not need another launcher command. Run them through
the normal interface one at a time:

```bash
./bin/tt-lang-sim --backend=emule examples/eltwise_add.py
./bin/tt-lang-sim --backend=emule examples/single_node_matmul.py
```

## Troubleshooting

### Docker is unavailable

Check the daemon and selected context:

```bash
docker info
docker context show
```

### The environment is not installed

Run the installer from the same checkout:

```bash
./scripts/install-tt-lang-emule.sh
```

The simulator intentionally does not build or download missing components while
executing a program.

### The compiler checkout changed

The installed compiler is tied to the exact TT-Lang source used during
installation. Re-run the installer after switching branches, pulling new
commits, or changing local source files.

### Emulator source access fails

Confirm that `TTLANG_EMULE_RUNTIME_SOURCE_URL` names the approved repository and
that the host can read it. The installer fetches only the exact commit from the
manifest and uses the host's Git credentials; it never substitutes a different
emulator revision.
