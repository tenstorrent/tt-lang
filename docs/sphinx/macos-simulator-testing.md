# macOS: Building and Simulator-Based Runtime Testing

This page covers developing tt-lang on an Apple Silicon Mac: building the
LLVM/MLIR toolchain natively, provisioning the hardware simulator, and running
tt-lang's runtime (`REQUIRES: tt-device`) tests against that simulator inside a
Linux VM. Runtime tests need tt-metal, which is Linux-only, so they run in a
Lima VM rather than on the macOS host.

## Two different "simulators"

Do not conflate them:

| | Functional simulator | Hardware simulator |
|---|---|---|
| Name | `tt-lang-sim` | ttsim -- `libttsim.so` (public `tenstorrent/ttsim`; custom forks supported) |
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

## The hardware simulator (`libttsim.so`)

tt-metal loads a `libttsim.so` (built for a specific chip) via
`TT_METAL_SIMULATOR`. The runtime-test setup below stages one automatically with
`test/hw-sim/vm-install-sim.sh`:

- **Default (public ttsim):** download a pinned
  [`tenstorrent/ttsim`](https://github.com/tenstorrent/ttsim) release for the chip
  and host architecture. ttsim publishes Linux/aarch64 assets (e.g.
  `libttsim_wh_aarch64.so`), so Apple Silicon needs no sim build. Pin the release
  with `TTSIM_VERSION` and pick the chip with `TTSIM_CHIP=wh|bh`.
- **Custom simulator:** to use an internal or locally-modified simulator, set
  `TTLANG_SIM_SO=<path>` to a prebuilt `libttsim.so`, or `TTLANG_SIM_SRC=<dir>` to
  a simulator source tree that the script builds with `./make.py`.

Either way it stages `libttsim.so` and the matching tt-metal SOC descriptor at
`/opt/ttlang-toolchain/sim/`. A downloaded release must be ABI-compatible with the
tt-metal built into the toolchain; see the
[libttsim API](https://github.com/tenstorrent/ttsim/blob/main/docs/libttsim_api.md).

### Building a custom simulator from source

A simulator source tree builds its `.so` with `./make.py` (what `TTLANG_SIM_SRC`
runs):

```bash
cd <simulator-src>
./make.py src/_out/release_wh/libttsim.so src/_out/release_bh/libttsim.so
```

- Build only the `.so` targets -- not `./make.py :build`, which pulls in
  `tests/:build` and requires the SFPI RISC-V cross-compiler (Linux-only).
- The `.so` is architecture- and OS-specific: build it where it runs (the aarch64
  VM). A macOS build is an arm64 Mach-O that only loads on macOS -- useful for a
  standalone compile/smoke check (`./make.py tests/_out/rv32_alu`, then
  `tests/_out/rv32_alu --sim src/_out/release_wh/libttsim.so --loops 0 1000`).
- Apple clang (`-Werror`) may flag `static inline` helpers that GCC (Linux) does
  not; annotate them `[[maybe_unused]]`. Such patches are macOS-host-only.

## Runtime tests in a Lima VM

[Lima](https://lima-vm.io/) runs an Ubuntu aarch64 VM natively on Apple Silicon.
tt-metal builds and runs there, so tt-lang's `REQUIRES: tt-device` tests execute
against a `libttsim.so` with no silicon. tt-mlir is **not** needed.

Everyday work stays in your single checkout (`~/tt/tt-lang`): tt-lang builds
directly from the mounted source into `build-lima/`, and tests run there. Only the
one-time **toolchain** build (LLVM + tt-metal) needs a VM-local scratch copy --
tt-metal writes firmware ELFs and its CPM cache into its own source tree, and
virtiofs rejects those writes -- and that copy is internal to
`vm-build-toolchain.sh` (reclaimed when it finishes); you never work in it.

Helper scripts live in the repo at `test/hw-sim/` (`ttsim-vm.yaml`,
`vm-build-toolchain.sh`, `vm-install-sim.sh`, `vm-build-ttlang.sh`,
`vm-resume-build.sh`). They discover the mount by finding the dir that contains
`tt-lang` (override with `SRC_HOST`).

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

Run from the tt-lang repo root on the host. tt-lang lives under a **TT root**
that the VM mounts -- default `~/tt`, set by the `mounts:` entry in
`test/hw-sim/ttsim-vm.yaml` (change it if your clone lives elsewhere).

```bash
# 1. Check out tt-metal and its submodules (umd, tracy) so the build finds them
#    (works even if the repo was cloned without --recurse-submodules).
git submodule update --init --recursive --depth 1 third-party/tt-metal

# 2. Create the VM (Ubuntu 24.04 aarch64). Config path is relative to the repo root.
limactl start --tty=false --name=ttlang-ttsim test/hw-sim/ttsim-vm.yaml

# Discover where the TT root (the dir holding tt-lang) is mounted in the guest:
TT="$(limactl shell ttlang-ttsim -- bash -c 'findmnt -nrt virtiofs -o TARGET | while read m; do [ -d "$m/tt-lang" ] && { echo "$m"; break; }; done')"

# 3. Build the toolchain (LLVM + tt-metal) into /opt/ttlang-toolchain. Detached, so
#    it survives a dropped SSH session; logs to the mount (host-visible).
limactl shell ttlang-ttsim -- bash -c \
  "setsid bash $TT/tt-lang/test/hw-sim/vm-build-toolchain.sh > $TT/tt-lang/toolchain-build.log 2>&1 </dev/null & disown"
#    watch on the host: tail -f toolchain-build.log

# 4. Install the simulator into the toolchain. Default: download a pinned public
#    ttsim release. For a custom simulator, prefix TTLANG_SIM_SRC=<dir> (build from
#    source) or TTLANG_SIM_SO=<path> (prebuilt); TTSIM_CHIP defaults to wh.
limactl shell ttlang-ttsim -- bash "$TT/tt-lang/test/hw-sim/vm-install-sim.sh"

# 5. Auto-set the simulator env for every interactive shell.
limactl shell ttlang-ttsim -- bash -c \
  'grep -q TT_METAL_SIMULATOR ~/.bashrc || printf "\nexport TT_METAL_SIMULATOR=/opt/ttlang-toolchain/sim/libttsim.so\nexport TT_METAL_SLOW_DISPATCH_MODE=1\n" >> ~/.bashrc'
```

### Build and test tt-lang (one checkout, `build-lima`)

Everything below uses your single tree; edit tt-lang on the host and it is live in
the VM via the mount. `limactl shell ttlang-ttsim` opens an interactive shell,
just like `docker exec -it`.

```bash
limactl shell ttlang-ttsim
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
  `vm-build-toolchain.sh`, rerun `vm-install-sim.sh` (bump `TTSIM_VERSION` if the
  new tt-metal needs a newer ttsim ABI), then rebuild `build-lima`.

Setting `TT_METAL_SIMULATOR` is what enables the runtime tests: tt-lang's
`test/lit.cfg.py` adds the `tt-device` lit feature when `TT_METAL_SIMULATOR` is set
(or when hardware is present), so `REQUIRES: tt-device` tests run under the
simulator. See [testing.md](testing.md).

## Reference: how tt-mlir CI uses the same simulator

tt-mlir's CI uses the same public-ttsim mechanism that `vm-install-sim.sh`
automates: download a prebuilt `libttsim_{wh,bh}.so` from `tenstorrent/ttsim`
releases (Linux/aarch64 assets are suffixed `_aarch64`, e.g.
`libttsim_wh_aarch64.so`), place it beside the matching tt-metal SOC descriptor
(`wormhole_b0_80_arch.yaml` / `blackhole_140_arch.yaml`) as `soc_descriptor.yaml`,
and set `TT_METAL_SIMULATOR` + `TT_METAL_SLOW_DISPATCH_MODE=1`. tt-mlir
additionally runs `ttrt query --save-artifacts` to produce `system_desc.ttsys`
and passes `--sys-desc` to its gated tests.

## Status

Verified end-to-end on a fresh Apple Silicon Lima VM (Ubuntu 24.04 aarch64):
current LLVM + tt-metal build from the submodules, tt-lang builds against that
toolchain in `build-lima`, and pytest / me2e / lit device tests (e.g.
`dram_interleaved_add`) execute under `TT_METAL_SIMULATOR` and pass -- against
both the default public ttsim release and a custom simulator built from source
via `TTLANG_SIM_SRC`.
