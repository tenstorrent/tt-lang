# macOS: Building and Simulator-Based Runtime Testing

This page covers developing tt-lang on an Apple Silicon Mac: building the
LLVM/MLIR toolchain natively, building the hardware simulator, and running
tt-lang's runtime (`REQUIRES: tt-device`) tests against that simulator inside a
Linux VM. Runtime tests need tt-metal, which is Linux-only, so they run in a
Lima VM rather than on the macOS host.

## Two different "simulators"

Do not conflate them:

| | Functional simulator | Hardware simulator |
|---|---|---|
| Name | `tt-lang-sim` | ttsim / `libttsim.so` (local fork: craq-sim) |
| What it is | tt-lang ops as pure Python (torch-backed) | Simulates the Tensix/RISC-V device; loaded by tt-metal via `TT_METAL_SIMULATOR` |
| Needs tt-metal? | No | Yes |
| Runs on macOS host? | Yes (native) | No -- needs Linux (Lima VM) |
| Docs | [simulator.md](simulator.md) | this page |

This page is about the hardware simulator. For `tt-lang-sim`, see
[simulator.md](simulator.md).

## macOS host toolchain build (LLVM/MLIR only)

On macOS, [`scripts/build-and-install.sh`](build.md) builds LLVM/MLIR and the
Python venv but skips tt-metal (Linux-only runtime dependencies). This is enough
for compile-only tests and MLIR lit tests on the host; runtime tests need the VM
(below).

```bash
# The default python3 may be too new for tt-metal/ttnn; use a supported one.
# Ensure the install prefix is writable (see build.md).
sudo mkdir -p /opt/ttlang-toolchain && sudo chown "$USER" /opt/ttlang-toolchain

./scripts/build-and-install.sh --toolchain-only --python "$(brew --prefix)/bin/python3.12"
```

Notes:
- Python 3.10+ is required (3.12 recommended); pin an installed supported
  interpreter with `--python` since the system `python3` is often newer than
  tt-metal/ttnn support.
- The macOS build produces an LLVM/MLIR-only toolchain (`mlir-opt`, `llvm-lit`,
  `MLIRConfig.cmake`, and the venv under `/opt/ttlang-toolchain`).

### macOS portability fixes in `build-and-install.sh`

Three changes make the script work correctly on macOS (`/bin/bash` 3.2, BSD
coreutils):
- `set -Eeo pipefail` + an `ERR` trap so any failure exits non-zero with a
  message (a failure can never be reported as success). Note a caller that pipes
  output through `tee` still masks the code -- invoke without a masking pipe.
- `df -BM` -> `df -h` (GNU-only flag; BSD `df` errors on `-BM`).
- The `do_finalize` normalize + cleanup steps run only on Linux. They are
  CI/Docker packaging (relocatable install; stub non-essential LLVM binaries to
  slim the image) and rely on bash 4 (`mapfile`, `declare -A`) and GNU tools
  (`chmod --reference`, `strip --strip-unneeded`); on macOS the local toolchain
  is used in place, so they are skipped.

## Building the hardware simulator (`libttsim.so`)

The simulator's `libttsim.so` is architecture- and OS-specific. Build it for the
platform where it will run.

### On the macOS host (standalone verification only)

Useful to confirm the simulator compiles and passes host-driven checks, but the
resulting `.so` is an arm64 Mach-O and only loads on macOS:

```bash
cd craq-sim   # your ttsim/craq-sim checkout
./make.py src/_out/release_wh/libttsim.so src/_out/release_bh/libttsim.so
```

- Build only the `.so` targets -- not `./make.py :build`, which pulls in
  `tests/:build` and requires the SFPI RISC-V cross-compiler (Linux-only).
- Host-native smoke test (no SFPI): `rv32_alu`/`rv64_alu` inject RISC-V
  instructions from the host and load the `.so`:
  ```bash
  ./make.py tests/_out/rv32_alu
  tests/_out/rv32_alu --sim src/_out/release_wh/libttsim.so --loops 0 1000
  ```
  `tensix_fpu`/`tensix_sfpu` and the `*.elf` tests are gated to Linux -- their
  on-device kernels are compiled with the SFPI cross-compiler.

Apple clang (`-Werror`) may flag `static inline` helpers that GCC (Linux) does
not; annotate them `[[maybe_unused]]` following the file's existing idiom. Such
patches are host-side only -- GCC in the VM does not need them.

### In the Lima VM (for the runtime-test loop)

The VM is Ubuntu aarch64, so build `libttsim.so` there (a macOS Mach-O will not
load under Linux) and stage it at `/opt/ttlang-toolchain/sim/libttsim.so`; the
one-time setup below does this.

## Runtime tests in a Lima VM

[Lima](https://lima-vm.io/) runs an Ubuntu aarch64 VM natively on Apple Silicon.
tt-metal builds and runs there, so tt-lang's `REQUIRES: tt-device` tests execute
against a craq-sim `libttsim.so` with no silicon. tt-mlir is **not** needed.

Everyday work stays in your single checkout (`~/tt/tt-lang`): tt-lang builds
directly from the mounted source into `build-lima/`, and tests run there. Only the
one-time **toolchain** build (LLVM + tt-metal) needs a VM-local scratch copy --
tt-metal writes firmware ELFs and its CPM cache into its own source tree, and
virtiofs rejects those writes -- and that copy is internal to
`vm-build-toolchain.sh` (reclaimed when it finishes); you never work in it.

Helper scripts live in the repo at `test/hw-sim/` (`craqsim-vm.yaml`,
`vm-build-toolchain.sh`, `vm-build-ttlang.sh`, `vm-resume-build.sh`). They discover
the mount by finding the dir that contains `tt-lang` (override with `SRC_HOST`).

**Prerequisites:** Homebrew, `brew install lima`, and ~80 GB free host disk. The
toolchain build tree lands on the VM's disk image, which draws from host free
space and is not reclaimed by deleting files inside the VM (see disk note).

### aarch64 gotchas the scripts handle

- tt-metal's `install_dependencies.sh` downloads an **amd64-only** `openmpi-ulfm`
  `.deb`; run it with `--no-distributed` (single-device sim needs no MPI).
- A wrong-arch **x86-64 `cmake`** in `/usr/local/bin` can shadow the apt arm64
  one; remove it.
- `install_dependencies.sh` installs Clang from apt.llvm.org as a versioned
  package (e.g. `clang-20`); tt-lang's CMake wants bare `clang`/`clang++`, so
  register the newest installed `clang-N` with `update-alternatives`.
- The toolchain build runs from a VM-local copy, not the mount: tt-metal writes
  into its own tree and those writes fail over virtiofs. tt-lang itself builds
  fine over virtiofs, so `build-lima` uses your real checkout directly.

### One-time setup

Run from the tt-lang repo root on the host. tt-lang and craq-sim sit under a
common **TT root** that the VM mounts -- default `~/tt`, set by the `mounts:` entry
in `test/hw-sim/craqsim-vm.yaml` (change it if your clones live elsewhere).

```bash
# 1. Check out tt-metal and its submodules (umd, tracy) so the build finds them
#    (works even if the repo was cloned without --recurse-submodules).
git submodule update --init --recursive --depth 1 third-party/tt-metal

# 2. Create the VM (Ubuntu 24.04 aarch64). Config path is relative to the repo root.
limactl start --tty=false --name=ttlang-craqsim test/hw-sim/craqsim-vm.yaml

# Discover where the TT root (the dir holding tt-lang) is mounted in the guest:
TT="$(limactl shell ttlang-craqsim -- bash -c 'findmnt -nrt virtiofs -o TARGET | while read m; do [ -d "$m/tt-lang" ] && { echo "$m"; break; }; done')"

# 3. Build the toolchain (LLVM + tt-metal) into /opt/ttlang-toolchain. Detached, so
#    it survives a dropped SSH session; logs to the mount (host-visible).
limactl shell ttlang-craqsim -- bash -c \
  "setsid bash $TT/tt-lang/test/hw-sim/vm-build-toolchain.sh > $TT/tt-lang/toolchain-build.log 2>&1 </dev/null & disown"
#    watch on the host: tail -f toolchain-build.log

# 4. Build craq-sim's libttsim.so (Linux aarch64) and stage it with a SOC descriptor.
limactl shell ttlang-craqsim -- bash -c "
  set -e
  cd $TT/craq-sim && ./make.py src/_out/release_wh/libttsim.so
  mkdir -p /opt/ttlang-toolchain/sim
  cp src/_out/release_wh/libttsim.so /opt/ttlang-toolchain/sim/libttsim.so
  cp /opt/ttlang-toolchain/tt-metal/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml \
     /opt/ttlang-toolchain/sim/soc_descriptor.yaml"

# 5. Auto-set the simulator env for every interactive shell.
limactl shell ttlang-craqsim -- bash -c \
  'grep -q TT_METAL_SIMULATOR ~/.bashrc || printf "\nexport TT_METAL_SIMULATOR=/opt/ttlang-toolchain/sim/libttsim.so\nexport TT_METAL_SLOW_DISPATCH_MODE=1\n" >> ~/.bashrc'
```

### Build and test tt-lang (one checkout, `build-lima`)

Everything below uses your single tree; edit tt-lang on the host and it is live in
the VM via the mount. `limactl shell ttlang-craqsim` opens an interactive shell,
just like `docker exec -it`.

```bash
limactl shell ttlang-craqsim
cd "$(findmnt -nrt virtiofs -o TARGET | while read m; do [ -d "$m/tt-lang" ] && { echo "$m"; break; }; done)/tt-lang"

# Build tt-lang from this source into build-lima/, against the prebuilt toolchain
# (once to configure, incremental thereafter). Or: bash test/hw-sim/vm-build-ttlang.sh
cmake -G Ninja -B build-lima -DTTLANG_USE_TOOLCHAIN=ON -DTTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain
cmake --build build-lima

source build-lima/env/activate    # venv + paths for this build
# TT_METAL_SIMULATOR + slow-dispatch are already set via ~/.bashrc

# Run tests (all against your checkout):
python -m pytest test/python/pipe/test_broadcast_2d.py -xvs   # a pytest
python -m pytest test/me2e/test_compute_ops.py -k add -xvs    # me2e
llvm-lit -v build-lima/test/python/dram_interleaved_add.py    # a python lit test
```

Remember the split: **pytest runs against `test/...`, lit against
`build-lima/test/...`** (the generated `lit.site.cfg.py` lives in the build dir, so
`llvm-lit test/...` on the source tree fails with `AttributeError: ...
python_executable`).

### Parallel test runs

`check-ttlang-pytest` and `check-ttlang-me2e` append `-n ${TTLANG_SIM_PYTEST_JOBS}`
to pytest when `TT_METAL_SIMULATOR` is defined **at configure time** -- the signal
that this build targets the simulator. An interactive shell exports it via
`~/.bashrc` and `vm-build-ttlang.sh` sets it explicitly, so configuring through
either enables parallelism; a bare `cmake` in a shell that lacks it configures the
targets serially. Each worker starts its own `libttsim.so` device (~2-3 GiB), so
the default of 2 suits the 16 GiB reference VM; raise it (with the VM's `memory:`)
via `-DTTLANG_SIM_PYTEST_JOBS=<N>`.

```bash
ninja -C build-lima check-ttlang-me2e     # runs pytest -n 2 under the sim
```

### Operational notes

- **Detach long builds** (`setsid ... & disown`) and log to the mount for host
  visibility; a dropped `limactl shell` otherwise orphans the build. The toolchain
  build is the long one; the incremental `build-lima` build is short.
- **Disk:** the VM diffdisk grows with the toolchain build and is not returned to
  the host when files are deleted inside the VM (`fstrim` did not reclaim it here).
  If it bloats across failed attempts, delete and recreate the VM.
- **Post-uplift:** after bumping the LLVM/tt-metal submodules, re-init tt-metal's
  submodules (step 1), remove `/opt/ttlang-toolchain` (toolchain reuse is keyed on
  file existence, not the submodule SHA -- see [build.md](build.md)), rerun
  `vm-build-toolchain.sh`, then rebuild `build-lima`.

Setting `TT_METAL_SIMULATOR` is what enables the runtime tests: tt-lang's
`test/lit.cfg.py` adds the `tt-device` lit feature when `TT_METAL_SIMULATOR` is set
(or when hardware is present), so `REQUIRES: tt-device` tests run under the
simulator. See [testing.md](testing.md).

## Reference: how tt-mlir CI uses the same simulator

tt-mlir's CI uses the identical mechanism and is a good template:
download a prebuilt `libttsim_{wh,bh}.so` from `tenstorrent/ttsim` releases,
rename it to `libttsim.so`, copy the matching tt-metal SOC descriptor
(`wormhole_b0_80_arch.yaml` / `blackhole_140_arch.yaml`) beside it as
`soc_descriptor.yaml`, set `TT_METAL_SIMULATOR{,_HOME}`,
`TT_METAL_SLOW_DISPATCH_MODE=1`, and `TT_METAL_DISABLE_SFPLOADMACRO=1`, run
`ttrt query --save-artifacts` to produce `system_desc.ttsys`, then run the gated
tests with `--sys-desc`. Those release binaries are Linux x86_64; on Apple
Silicon build `libttsim.so` for Linux aarch64 in the VM instead.

## Status

Verified end-to-end on Apple Silicon:
- macOS host: the LLVM/MLIR toolchain builds (exit 0); craq-sim `libttsim.so`
  builds for wormhole/blackhole and the `rv32_alu` host smoke passes.
- Lima VM (Ubuntu 24.04 aarch64): current LLVM + tt-metal build from the
  submodules, tt-lang builds against that toolchain in `build-lima`, craq-sim
  `libttsim.so` builds for Linux aarch64, and pytest / me2e / lit device tests
  (e.g. `dram_interleaved_add`) execute on the simulated wormhole device under
  `TT_METAL_SIMULATOR` and pass.
