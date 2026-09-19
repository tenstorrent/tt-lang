---
name: uplift
description: Uplift the LLVM and tt-metal submodule pins, validate the result on hardware, and tell an uplift regression apart from an environment failure. Use when asked to uplift or bump submodules, move TT_METAL_TAG, or debug a build or device failure that appeared after a pin change.
---

# Uplifting LLVM and tt-metal

`docs/sphinx/build.md`, section "Uplifting Submodules", is the canonical
procedure: how to edit `third-party/tt-metal-version`, the two-phase split
between a PyPI-aligned pin and an S3-only pin, the rebuild commands, and the
tagging rules. Read it first. This skill covers what that document does not:
how to validate an uplift on hardware, and how to attribute a failure.

## Choosing pins

- Prefer a tt-metal tag that downstream repositories already pin. Uplifting to
  a tag nothing else runs means finding its bugs first.
- `TTSIM_VERSION` in `test/hw-sim/vm-install-sim.sh` must match
  `tt_metal/ttsim-version` at the chosen tt-metal tag.
- Check every patch in `third-party/patches` applies at the new tag before
  starting a long build.

## Building

- Toolchain reuse is keyed on file existence, not on the recorded SHA. When a
  submodule SHA changes, delete both the toolchain build directory and the
  target toolchain directory, or the old LLVM and tt-metal are silently kept
  and the new pins are never compiled.
- Expect LLVM API churn in tt-lang. Enumerate every break in one pass with
  `cmake --build <dir> -- -k 0` instead of one rebuild per error. `-Werror`
  turns upstream deprecations into build failures, so a deprecated ODS builder
  overload is a break like any other.
- A configured tree already has the patches applied, because `BuildTTMetal`
  applies them at configure time. Forward `git apply --check` then fails;
  `git apply --check --reverse` confirms the patch is present.

## Validating on hardware

- Both hardware runners go through `call-test-hardware.yml` and run the same
  phases, so `test/python` and `test/me2e` run in full on n150 as well as on
  BH-Quietbox-2. The n150 job carries `continue-on-error` so the whole matrix
  finishes before a verdict, but `check-hardware` then reads every hardware
  job's result and fails the run on it. An n150 failure is as real as a
  Quietbox one.
- For an A/B against a baseline, build both variants in their own IRD images on
  one exclusive host and run the same test. A Blackhole Quietbox reserved
  through Slurm matches the CI runner. Reserve it, run the device ownership
  preflight, and release it afterwards.
- A test that never returns is usually a device hang, not a slow test. Run
  pytest with `--timeout-method=thread`, as the hardware harness does: it
  terminates a process stuck in a C-level device call, where SIGALRM cannot.
  It still cannot fire while the blocking call holds the GIL, which leaves the
  timeout thread unable to run, so wrap a manual run in an outer `timeout` too.
  Identify the test with `py-spy dump --pid <pid>` and get the native stack
  with `gdb -p <pid> -batch -ex 'thread 1' -ex 'bt 25'`.
- A device hang leaves the board unusable, reporting an active ethernet core
  heartbeat timeout on the next device open. Recover with `tt-smi -r 0,1,2,3`
  and delete `~/.cache/tt_metal_*`. Reset only after confirming no other
  process holds a device, and never to clear someone else's workload.
- `fuser` and `lsof` run as your user and cannot see device holders owned by
  root or running inside another user's container, so they report devices as
  free when they are not. Check with
  `sudo -n grep -l tenstorrent /proc/[0-9]*/maps`.

## Attributing a regression

- Compare against the same source on the previous toolchain before concluding
  the uplift caused a failure. CI history on `main` at the branch point is the
  cheapest baseline.
- To separate an LLVM regression from a tt-metal one, compare everything the
  compiler produced, which is more than the kernels. tt-lang writes generated
  kernels to `/tmp/$USER/ttlang_kernel_*.cpp`, named by a hash of their
  content, and derives descriptor metadata (argument specs, DFB indices, fabric
  routes, compute configuration) separately; `ttl_api.py` shares a descriptor
  only when both the C++ and that metadata match. Run the failing test under
  each toolchain with `TTLANG_COMPILE_ONLY=1`, which generates kernels without
  launching them, and compare the kernels and `TTLANG_FINAL_MLIR`. Only when
  both are identical did the compiler output not change, leaving the regression
  in the tt-metal runtime, LLK or firmware.

## Carrying an upstream fix

When an upstream fix is needed but not yet merged, carrying it is preferable to
pinning backwards.

- Put the upstream diff in `third-party/patches/ttmetal-*.patch` with a header
  naming the pull request it backports and the condition for deleting it.
- The toolchain cache key includes the tree hash of `third-party/patches`, so
  adding or editing a patch rebuilds the toolchain. Keep it that way: without
  it a patch added without a submodule bump restores a stale toolchain and
  never takes effect.
- `third-party/patches` is excluded from the `trailing-whitespace` and
  `end-of-file-fixer` pre-commit hooks, because a unified diff writes a blank
  context line as a single space and trimming it stops the patch applying.

## CI sequencing

- Toolchain caches are scoped per ref. A pull request run reads its own merge
  ref, its base branch and `main`. A cache built by a dispatch on a branch ref
  is not visible to that branch's pull request run.
- `ci.yml` runs only for pull requests targeting `main`, so a stacked pull
  request gets no CI, and no toolchain cache, until it retargets `main`.
- `ci.yml` cancels in-progress runs on the same ref. Let the toolchain cache
  save before pushing again to a branch whose toolchain is still building, or
  the build restarts from scratch.
- The image build downloads tt-metal's `install_dependencies.sh`, which fetches
  from `apt.llvm.org`. That host fails intermittently and takes the whole image
  build with it, which skips every hardware job. Re-run before investigating.
