# Getting started with compiler-backed emulation

The `emule` backend compiles TT-Lang programs and executes their generated
kernels through tt-metal and tt-emule inside Docker. Install the environment
once, then run programs with `./bin/tt-lang-sim --backend=emule`.

Relative command paths in this guide assume the TT-Lang checkout root as the
working directory, except where a container shell is specified.

The repository provides a pinned environment through
`config/tt-lang-emule-stack.json`. This manifest records the required compiler
baseline, tt-emule revision, tt-metal revision, base image, and P150 target.
Installation builds the current TT-Lang checkout, which must contain that
compiler baseline.

## Host prerequisites

- Git and Python 3.10 or newer on `PATH`.
- A Docker-compatible daemon, running and accessible without `sudo`. The
  [Docker installation guide](https://docs.docker.com/get-started/get-docker/)
  covers Docker Desktop on macOS and Docker Engine on Linux.
- Support for `linux/amd64` containers: tt-emule JITs x86-64 shared objects.
  On Apple Silicon, see Docker Desktop's
  [virtualization and Rosetta settings](https://docs.docker.com/desktop/settings-and-maintenance/settings/#general)
  for x86-64 emulation support and acceleration.
- The approved tt-emule repository URL and Git access to that repository. The
  installer uses the host's existing Git credentials.

Check the host tools before installation:

```bash
git --version
python3 --version
docker info
```

## Obtain TT-Lang

Start from a TT-Lang source checkout containing the compiler-backed backend:

```bash
git clone https://github.com/tenstorrent/tt-lang.git
cd tt-lang
```

The installer uses the container's prebuilt LLVM toolchain and pinned tt-metal
source, so the checkout's LLVM and tt-metal submodules can remain uninitialized.
Keep the TT-Lang Git history available for the compiler baseline ancestry check.

The installer validates that the checkout contains the compiler baseline
recorded in the stack manifest. It also fetches the exact
tt-emule revision and verifies that tt-emule pins the same tt-metal revision as
the manifest.

## Install the environment

Set the emulator repository location supplied by the tt-emule owners, then run
the installer from the checkout root:

```bash
export TTLANG_EMULE_RUNTIME_SOURCE_URL=REPOSITORY_URL
./scripts/install-tt-lang-emule.sh
```

The URL supplies the source for the manifest's pinned emulator commit. The
installer uses the recorded emulator, tt-metal, and base image together in a
Linux/x86-64 environment and verifies their source revisions before building.

Installation builds the pinned tt-emule/tt-metal Docker image and compiles this
TT-Lang checkout into a persistent Docker volume. It can take substantial time,
CPU, memory, and disk space on its first run. The installer prints the runtime
image, compiler build volume, and runtime cache volume names. On success it also
prints the compiler commit for which the environment was installed.

Installation prepares the compiler before the first program run. Subsequent
`tt-lang-sim` runs reuse the installed environment, which Docker retains in its
image and volume caches. Run the installer again after changing compiler
commits or build inputs, or to restore a missing image or incomplete compiler
environment. The launcher checks the installation before executing a program
and reports when reinstallation is needed. The compiler build and the tt-metal
and tt-emule JIT caches live in named Docker volumes.

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

The launcher mounts the checkout and the current working directory. Use relative
paths for program inputs beneath the current working directory. Absolute paths
are passed unchanged and must refer to locations visible inside the container.

The launcher can also run from another working directory through its absolute
path. In this example, `program.py` is relative to the current directory:

```bash
/path/to/tt-lang/bin/tt-lang-sim --backend=emule program.py
```

To run programs directly on the host with the default Python backend, first
install and activate the environment described in
[Python backend setup](simulator.md#setup). The emule installer prepares the
container environment; native Python simulation uses its own host dependencies.
From the activated host environment:

```bash
./bin/tt-lang-sim program.py
./bin/tt-lang-sim --backend=python program.py
```

## Run tests with the existing test framework

Select tests and generate reports with TT-Lang's CMake, pytest, and lit commands
from an activated compiler build environment.

### Enter the installed Docker environment

Open a shell with the same installed compiler, runtime settings, mounts, and
working directory used by the emulator launcher:

```bash
./scripts/shell-tt-lang-emule.sh
```

The helper verifies the installed compiler and activates its environment before
starting Bash. The compiler build is available at `/ttlang-build`, and the source
checkout at `/workspace`. Linked worktrees use the same Git metadata mount as
program execution.

### Select tests

Inside the activated container shell, use the commands in [Testing](testing.md)
or the detailed
[`test/TESTING.md` guide](https://github.com/tenstorrent/tt-lang/blob/main/test/TESTING.md#running-tests).
Those references cover full suites, individual cases, pytest filtering, lit,
and report locations. Apply the installed environment's paths:

- Use `/ttlang-build` wherever the testing instructions use `build`.
- For direct pytest invocations, pass `-c /ttlang-build/test/pytest.ini` to load
  the installed build's generated configuration.
- Select Python lit cases under `/ttlang-build/test/python`, which contains
  their configured test environment.

A separate native Linux build uses its own build directory. Compiler-only
tests exercise compiler behavior; device execution tests exercise tt-emule in
the installed Linux environment.

## Validate and inspect the environment

The installer validates the current TT-Lang checkout against the manifest's
compiler baseline. It also verifies the emulator checkout commit, the P150
descriptor, and the emulator's exact tt-metal pin before building. Run the same
checks directly with:

```bash
python3 scripts/tt-lang-emule-stack.py \
  --manifest config/tt-lang-emule-stack.json \
  validate --compiler-source . --emulator-source /path/to/emulator
```

Every built image records its resolved inputs as OCI labels and in
`/opt/tt-emule-runtime/stack.json`. The original supported-stack manifest is
stored beside it as `source-manifest.json`, and its SHA-256 is verified while
the image is built. These records identify the supported manifest and exact
runtime inputs used to build the image. Inspect an artifact without running a
workload with:

```bash
docker image inspect tt-lang-emule:TAG \
  --format '{{json .Config.Labels}}'
docker run --rm --entrypoint cat tt-lang-emule:TAG \
  /opt/tt-emule-runtime/stack.json
```

## Known limitations

The supported target is a single emulated Blackhole P150 device with the full,
unharvested 13x10 compute grid. The launcher selects the emulator's P150
descriptor and configures tt-metal's hybrid allocator before opening the
device. Complete models and multi-device workloads require further validation.

Known compiler-suite failures with the pinned runtime include RISC-V inline
assembly rejected by the x86 JIT, a missing RMSNorm SFPU header, incorrect results
in some dynamic-buffer reuse and collective tests, and missing DPRINT output.
The incorrect-output and DPRINT failures still require isolated reproducers;
they are not all established tt-emule defects. Device execution tests provide
the evidence for emulated execution; compiler-only tests cover the compiler
itself.

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

The installer prepares the image and compiler build for subsequent program runs.

### The compiler checkout changed

The installed compiler is tied to the exact TT-Lang source used during
installation. Re-run the installer after switching branches, pulling new
commits, or changing compiler/build inputs. The installed compiler can be reused
while editing workload scripts and generating output files.

### Emulator source access fails

Confirm that `TTLANG_EMULE_RUNTIME_SOURCE_URL` names the approved repository and
that the host can read it. The installer uses the host's Git credentials to
fetch the manifest's exact emulator commit.
