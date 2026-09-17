# Getting started with Docker simulation

The `emule` backend compiles TT-Lang programs and executes the generated kernels
through tt-metal and tt-emule inside Docker. The launcher prepares the runtime,
builds the compiler, and activates the container environment automatically.
Tenstorrent hardware, a host compiler build, and host Python package installation
are not required.

The separate `python` backend remains the lightweight option for kernel logic
and Python debugging. Its installation is described in
[Simulation backends](simulator.md#setup); it does not require Docker.

## Host prerequisites

- Git and Python 3.10 or newer on `PATH`.
- Docker Desktop on macOS, or Docker Engine on Linux, with the daemon running
  and accessible to the current user without `sudo`.
- Support for `linux/amd64` containers. Native x86-64 Linux executes this runtime
  directly. Apple Silicon requires x86-64 emulation in Docker Desktop; enabling
  Rosetta support reduces translation overhead where available.
- Access to the approved emulator source repository, or a compatible runtime
  image supplied by a maintainer. Source setup uses the host's Git credentials.
  Initial builds also require access to the base image and dependency downloads.

Check the host tools before setup:

```bash
git --version
python3 --version
docker info
```

If `python3` selects an older system interpreter, select an installed compatible
interpreter for the launcher. For example:

```bash
export TTLANG_EMULE_HOST_PYTHON=python3.12
```

This selection applies to the host launcher. The container supplies its own
Python environment and compiler tools.

## Obtain the source checkout

These commands describe development branch `kostas/tt-lang-sim-simple-cli`.
The branch is currently local development work and has not been published to
the upstream repository. An existing checkout of this branch, or access to a
maintainer-provided copy, is required until it is published. Installing the
PyPI simulator or cloning upstream `main` does not provide this interface.

An accessible local development checkout can be cloned with:

```bash
git clone --branch kostas/tt-lang-sim-simple-cli \
  /path/to/tt-lang-development-checkout tt-lang
cd tt-lang
```

Replace the source path with the supplied checkout. Once the branch is
published to the upstream repository, the equivalent network checkout is:

```bash
git clone --branch kostas/tt-lang-sim-simple-cli \
  https://github.com/tenstorrent/tt-lang.git
cd tt-lang
```

All remaining commands run from the checkout root and use
`./bin/tt-lang-sim`. The Docker workflow provides the compiler dependencies;
it does not require a separate host compiler build or environment activation.

## Set up the runtime once

Source setup fetches the exact emulator commit in
`config/tt-lang-emule-stack.json` and builds the corresponding Docker runtime.
Replace `REPOSITORY_URL` with the approved repository URL accessible through
the host's Git credentials:

```bash
./bin/tt-lang-sim --backend=emule --setup \
  --source-url REPOSITORY_URL --jobs 8
```

The stack manifest records the compiler baseline, emulator and tt-metal commits,
base image, and emulated device. Source setup checks the emulator revision and
its tt-metal pin against that manifest. An arbitrary latest checkout is not a
substitute for the recorded commit.

An existing Git checkout at that exact emulator commit can be used instead:

```bash
./bin/tt-lang-sim --backend=emule --setup \
  --source /path/to/emulator --jobs 8
```

If a compatible runtime image is already available, source building can be
replaced with image selection:

```bash
./bin/tt-lang-sim --backend=emule --setup --image REGISTRY/IMAGE:TAG
```

The image must contain the toolchain, emulator runtime, and entrypoint provided
by this project's emulator Dockerfile. Setup reuses an existing local image;
if Docker confirms that it is absent, setup attempts to download the named
image. There is currently no default published simulator image. A local image
tag does not imply that the same name is downloadable from a registry.

Setup builds TT-Lang as needed and runs the smoke test. Only successful setup
saves the selected runtime and build parallelism in `.ttlang-sim/emule.json`.
That file is local to this checkout and ignored by Git. Repeating setup replaces
the saved selection only after another successful smoke test.

The first source build can take a long time and use substantial CPU, memory,
and disk space. Docker retains the runtime image, compiler build, and kernel
caches for later commands. `--jobs 8` limits compiler build parallelism; a lower
value reduces peak build memory use. It can also be changed without selecting
another runtime:

```bash
./bin/tt-lang-sim --backend=emule --setup --jobs 4
```

## Run a program

After setup, the launcher loads the saved runtime automatically:

```bash
./bin/tt-lang-sim --backend=emule examples/eltwise_add.py
```

The program imports the real `ttl` and `ttnn` packages inside the container.
The compiler generates the kernels, and tt-emule executes them on an emulated
Blackhole P150 device. The example checks its output against a Torch reference.

Arguments belonging to the program follow `--`:

```bash
./bin/tt-lang-sim --backend=emule program.py -- --program-option value
```

The launcher mounts the source checkout and the program's working directory.
Absolute host paths supplied as program arguments are not rewritten into
container paths; relative paths under the mounted working directory are the
simplest way to provide program inputs.

Run the small compiler-to-emulator acceptance check at any time:

```bash
./bin/tt-lang-sim --backend=emule --smoke-test
```

This compiles and executes the external C++ call example and verifies its
tensor result. A successful smoke test confirms that the selected stack works
for that program; broader coverage is checked separately.

## Run examples and tests

The reference example set exercises addition, matrix multiplication, reduction,
and fused matrix multiplication with bias:

```bash
./bin/tt-lang-sim --backend=emule --examples
```

The current reduction example supplies rank-one and scalar tensors that the
compiler rejects with `Tensors must have at least 2 dimensions`. This is an
existing example/compiler mismatch, before emulator kernel execution. The
command reports the failure and continues through the remaining examples.

A short initial test selection checks compiler passes, Python bindings, and
packaging:

```bash
./bin/tt-lang-sim --backend=emule --test \
  --suite mlir --suite bindings --suite packaging
```

Run all six compiler suites with:

```bash
./bin/tt-lang-sim --backend=emule --test
```

The available suite names are:

| Suite | Coverage |
|---|---|
| `mlir` | Compiler IR, dialect, and transformation tests. |
| `bindings` | Python compiler binding tests. |
| `packaging` | Packaging and launcher tests. |
| `pytest` | Python compiler and device tests. |
| `me2e` | Compiler stages and device execution tests. |
| `python-lit` | Python lowering checks and selected device execution tests. |

The first three suites do not execute kernels in tt-emule. The remaining suites
mix compilation, rejection checks, and runtime execution; their total test
counts are not counts of emulator kernel launches. The full sweep currently
has failures and does not establish that every compiler workload is supported.
It excludes the Python simulator's `test/sim` suite and the tutorial suite.
Execution can take substantially longer on an emulated x86-64 Mac environment
than on native x86-64 Linux.

The test command continues after a failed suite and returns nonzero if any
suite fails. Every invocation creates a new report directory beneath
`.ttlang-sim/reports/` and prints its path. After the container starts the test
runner, that directory contains suite logs, fresh JUnit reports where produced,
and `summary.json`. `invocation.json` records the host checkout and requested
suites; the summary also records available runtime provenance. A failure during
image preparation or compiler building can prevent test reports from being
produced.

Select another report parent directory or repeat `--suite` to narrow a run:

```bash
./bin/tt-lang-sim --backend=emule --test \
  --suite bindings --reports-dir ./test-results
```

## Troubleshooting

### Docker is unavailable or cannot inspect the selected image

Check the daemon and context from the same terminal used for the launcher:

```bash
docker info
docker context show
docker image inspect REGISTRY/IMAGE:TAG
```

Replace the image name with the saved selection in `.ttlang-sim/emule.json`.
Different Docker contexts can point to different daemons and therefore different
local image stores. Image inspection errors are reported with their diagnostics;
an inspection failure is not treated as proof that an image must be downloaded.

### Image download is denied

A missing local image must either be available from the selected registry or
be rebuilt from emulator source. For a private published image, authenticate
to that registry with `docker login REGISTRY`. For a local-only tag, use
`--setup --source-url REPOSITORY_URL` or `--setup --source /path/to/emulator`
instead. The setup options select one runtime source at a time.

### Source access or pin validation fails

The source URL must be accessible using the host's Git authentication, and the
repository must contain the full emulator commit recorded in the stack manifest.
For local source setup, the checkout's `HEAD` and its `tt-metal-pin.txt` must
match the manifest. Setup reports the mismatch instead of selecting a different
revision automatically. Source repository access is required even though the
compiler and emulator execute in Docker.

### A saved setting appears to be ignored

Explicit `TTLANG_EMULE_*` environment settings take precedence over saved
configuration. The one-run `--runtime-image IMAGE` launcher option also
overrides the saved runtime. A shell carrying overrides from an earlier
experiment can therefore select a different runtime than `.ttlang-sim/emule.json`.
The Python backend does not read this emulator configuration.

See [Simulation backends](simulator.md#updating-the-supported-stack) for stack
validation, candidate workflows, and advanced compatibility overrides.
