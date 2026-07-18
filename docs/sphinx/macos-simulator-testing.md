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

./scripts/build-and-install.sh --toolchain-only --python "$(brew --prefix)/bin/python3.11"
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
load under Linux) and place it at the path the run harness expects
(`/var/tmp/sim/libttsim.so`, below).

## Runtime tests in a Lima VM

[Lima](https://lima-vm.io/) runs an Ubuntu aarch64 VM natively on Apple Silicon.
tt-metal builds and runs there, so tt-lang's `REQUIRES: tt-device` tests execute
against a craq-sim `libttsim.so` with no silicon. tt-mlir is **not** needed.

This is the verified flow: fresh VM -> current LLVM + tt-metal from the submodules
-> tt-lang -> craq-sim `libttsim.so` -> a passing device test. Helper scripts that
encode the aarch64 fixes live in the repo at `test/hw-sim/` (`craqsim-vm.yaml`,
`vm-build-toolchain.sh`, `vm-build-ttlang.sh`, `vm-resume-build.sh`). They run
inside the VM against the mounted source, auto-detecting the virtiofs mount that
contains `tt-lang` (override with `SRC_HOST`).

**Prerequisites:** Homebrew, `brew install lima`, and ~80 GB free host disk. The
LLVM + tt-metal build tree lands on the VM's disk image, which draws from host
free space and is not reclaimed by deleting files inside the VM (see disk note).

### aarch64 gotchas the scripts handle

- tt-metal's `install_dependencies.sh` downloads an **amd64-only** `openmpi-ulfm`
  `.deb`; run it with `--no-distributed` (single-device sim needs no MPI).
- A wrong-arch **x86-64 `cmake`** in `/usr/local/bin` can shadow the apt arm64
  one; remove it.
- apt installs versioned `clang-20` only, but tt-lang's CMake wants bare
  `clang`/`clang++` -- add them with `update-alternatives`.
- Build from a **VM-local copy** of the source, not the virtiofs mount: tt-metal
  writes into its own tree (CPM cache, firmware ELFs) and those writes fail over
  virtiofs.

### Build and run

Run from the tt-lang repo root on the host. tt-lang and craq-sim are assumed to
sit under a common **TT root** that the VM mounts -- default `~/tt`, set by the
`mounts:` entry in `test/hw-sim/craqsim-vm.yaml` (change it if your clones live
elsewhere). Host-side commands below are relative to the repo root; guest-side
paths are discovered from the mount.

```bash
# 1. Init tt-metal's submodules (umd, tracy) so the build finds them.
git -C third-party/tt-metal submodule update --init --depth 1

# 2. Create the VM (Ubuntu 24.04 aarch64). Config path is relative to the repo root.
limactl start --tty=false --name=ttlang-craqsim test/hw-sim/craqsim-vm.yaml

# Discover where the TT root (the dir holding tt-lang) is mounted in the guest:
TT="$(limactl shell ttlang-craqsim -- bash -c 'findmnt -nrt virtiofs -o TARGET | while read m; do [ -d "$m/tt-lang" ] && { echo "$m"; break; }; done')"
HW="$TT/tt-lang/test/hw-sim"

# 3. Provision + build the toolchain (LLVM + tt-metal) and install tt-metal into
#    it, from a VM-local source copy. Detached so it survives a dropped SSH session.
limactl shell ttlang-craqsim -- bash -c \
  "setsid bash $HW/vm-build-toolchain.sh >/var/tmp/toolchain-build.log 2>&1 </dev/null & disown"
#    watch: limactl shell ttlang-craqsim -- tail -f /var/tmp/toolchain-build.log

# 4. Build tt-lang against the toolchain.
limactl shell ttlang-craqsim -- bash "$HW/vm-build-ttlang.sh"

# 5. Build craq-sim's libttsim.so (Linux aarch64), stage it with the SOC descriptor.
limactl shell ttlang-craqsim -- bash -c '
  cd /var/tmp/craq-sim && ./make.py src/_out/release_wh/libttsim.so
  mkdir -p /var/tmp/sim && cp src/_out/release_wh/libttsim.so /var/tmp/sim/libttsim.so
  cp /opt/ttlang-toolchain/tt-metal/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml \
     /var/tmp/sim/soc_descriptor.yaml'

# 6. Run a runtime test under the simulator.
limactl shell ttlang-craqsim -- bash -c '
  cd /var/tmp/tt-lang && source build/env/activate
  export TT_METAL_SIMULATOR=/var/tmp/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1
  python3 test/python/dram_interleaved_add.py'
# -> "PASS: DRAM interleaved direct access works!"
#    ("EmulationDriver ... KHz" in the log confirms craq-sim drove the device.)
```

VM paths: source copies under `/var/tmp/{tt-lang,craq-sim}`, build dir
`/var/tmp/build-toolchain`, toolchain `/opt/ttlang-toolchain`, sim `/var/tmp/sim/`.

### Running tests interactively

`limactl shell ttlang-craqsim` opens an interactive shell in the VM, just like
`docker exec -it`. Append the two simulator `export`s to the VM's `~/.bashrc` so
every interactive shell sets them automatically; `source build/env/activate` is
still per-shell (venv + paths). Then run any suite:

```bash
limactl shell ttlang-craqsim              # interactive shell in the VM
cd /var/tmp/tt-lang
source build/env/activate
export TT_METAL_SIMULATOR=/var/tmp/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1

# pytest -- test/python/test_*.py and me2e (run against test/):
python -m pytest test/python/pipe/test_broadcast_2d.py -xvs
python -m pytest test/me2e/test_compute_ops.py -k add -xvs

# python lit test -- run against the build-configured tree, NOT the source tree:
llvm-lit -v build/test/python/dram_interleaved_add.py

# a lit test file as a plain script:
python3 test/python/dram_interleaved_add.py
```

Remember the split: **pytest runs against `test/...`, lit against `build/test/...`**.
`llvm-lit test/...` on the source tree fails (`AttributeError: ... python_executable`)
because the generated `lit.site.cfg.py` lives in the build dir.

### Iterating on tt-lang from the host tree (`build-lima`)

The steps above build tt-lang from a VM-local copy (`/var/tmp/tt-lang`), so host
edits don't appear there. For an edit-on-host / build-and-test-in-VM loop, build
tt-lang directly from the **mounted** source into a separate `build-lima/` dir
(tt-lang -- unlike tt-metal -- builds fine over virtiofs). The prebuilt toolchain is
reused, so this compiles only tt-lang's own dialects/bindings.

```bash
# Discover the mounted TT root (dir holding tt-lang), then point at tt-lang:
TT="$(limactl shell ttlang-craqsim -- bash -c 'findmnt -nrt virtiofs -o TARGET | while read m; do [ -d "$m/tt-lang" ] && { echo "$m"; break; }; done')"
TTLANG="$TT/tt-lang"

# Configure once, against the prebuilt toolchain:
limactl shell ttlang-craqsim -- bash -c \
  "cd $TTLANG && cmake -G Ninja -B build-lima -DTTLANG_USE_TOOLCHAIN=ON -DTTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain"

# After editing tt-lang on the host: rebuild (incremental) and test.
limactl shell ttlang-craqsim -- bash -c "cd $TTLANG && cmake --build build-lima"
limactl shell ttlang-craqsim -- bash -c "
  cd $TTLANG && source build-lima/env/activate
  export TT_METAL_SIMULATOR=/var/tmp/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1
  python -m pytest test/python/pipe/test_broadcast_2d.py -xvs"
```

`build-lima/` lives in your tt-lang checkout on the host, so its
`build.log` and artifacts are visible on the host. Use `source
build-lima/env/activate` (not `build/...`) for this build. The `/var/tmp` copy is
only needed for the one-time toolchain build (LLVM + tt-metal can't build over
virtiofs).

### Operational notes

- **Detach long builds** (`setsid ... & disown`) and have them write an exit-code
  marker; a dropped `limactl shell` otherwise orphans the build. Mirror the VM
  log into the mounted tree for host visibility
  (`limactl shell <vm> -- tail -F <vmlog> > <mounted-tree>/...`).
- **Disk:** the VM diffdisk grows with the build and is not returned to the host
  when files are deleted inside the VM (`fstrim` did not reclaim it here). If it
  bloats across failed attempts, delete and recreate the VM to reclaim the space.
- **Post-uplift:** after bumping the LLVM/tt-metal submodules, re-init tt-metal's
  submodules (step 1) and rebuild with a *clean* toolchain -- pass `--force-rebuild`
  to `build-and-install.sh` (or `rm -rf /opt/ttlang-toolchain` first), because
  toolchain reuse is keyed on file existence, not the submodule SHA (see
  [build.md](build.md)).

Setting `TT_METAL_SIMULATOR` is what enables the runtime tests: tt-lang's
`test/lit.cfg.py` adds the `tt-device` lit feature when `TT_METAL_SIMULATOR` is
set (or when hardware is present), so `REQUIRES: tt-device` tests run under the
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
  submodules, tt-lang builds against that toolchain, craq-sim `libttsim.so` builds
  for Linux aarch64, and `test/python/dram_interleaved_add.py` executes on the
  simulated wormhole device under `TT_METAL_SIMULATOR` and passes.
