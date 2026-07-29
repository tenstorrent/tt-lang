# Demo Integration Branch

Branch: `bnorris/demo`.

Base: `origin/main` at `816df6269c16` (`[ci][build] Consolidate manylinux wheel builds (#773)`).

Backup before this refresh: `backup/demo-before-refresh-20260729-071352`
at `ed4701c65e9e`.

Backup before #734 integration: `backup/demo-before-pr734-20260729-112617`
at `ce0b646c4a8d`.

Purpose: disposable aggregate branch for demos and integration testing. This
branch is not a PR review layer and must not become a parent for source PRs.

## Policy

- Maintain the source PR branches independently with Graphite.
- Maintain `bnorris/demo` with explicit `git merge --no-ff` commits from live
  `origin/bnorris/...` source tips.
- Rebuild `bnorris/demo` from `origin/main` when source branches rebase or
  force-push.
- Do not use cherry-picks as the default update mechanism. If an emergency demo
  repair uses a cherry-pick, record it here and replace it with a source-branch
  merge on the next rebuild.
- Keep this file updated with the base SHA, source tips, merge commits,
  exclusions, conflict notes, source PR follow-ups, and validation.

## Source Of Truth

- Current branch state is the live `origin/bnorris/...` refs in this checkout.
- Rebase status context: `/home/bnorris/tt/tt-lang-acc-3-defer-intermediate-dfbs/_updates.md`.
- PR inventory context: `/home/bnorris/tt/PRs.md`. Some entries may be stale,
  but the Pipes chain in that file is authoritative for which active Pipes PRs
  belong in this aggregate branch unless listed as excluded below.

## Included Inputs

| Order | PR | Source branch | Tip | Merge commit | Notes |
|---:|---:|---|---|---|---|
| 1 | 704 | `bnorris/3-defer-intermediate-dfbs` | `a4b521fbf52b` | `e047795b5` | Deferred compiler-created intermediate DFB materialization. |
| 2 | 778 | `bnorris/dfb-logical-physical-contract` | `362f0219e8c7` | `8f3493d91` | Logical DFB identity and physical allocation contract. |
| 3 | 775 | `bnorris/user-dfb-reuse` | `f354d885a581` | `58bda800d` | Concurrent DFB lifetime analysis and physical index reuse. |
| 4 | 733 | `bnorris/4-tensor-recurrence-scopes` | `80c0c07d0a5d` | `74f897f0a` | Tensor recurrence lowering through DST accumulation scopes. |
| 5 | 673 | `bnorris/support-indexed-pipenets` | `713892fa81df` | `076079b9c` | Indexed PipeNet receiver expressions. |
| 6 | 759 | `bnorris/ci-s3-publish-disk-caching` | `b3213d856cf3` | `562e10056`, `e6a7f24ec` | S3 publish disk caching and manylinux workflow updates. |
| 7 | 782 | `bnorris/pr700-1-pipe-schedule-validation` | `7309198cd219` | `273d28f95` | PipeNet schedule validation split. |
| 8 | 783 | `bnorris/pr700-2-computed-pipe-addresses` | `912573824cca` | `b9b68d68d` | Computed PipeNet receiver addresses. |
| 9 | 765 | `bnorris/pr700-counter-scaling` | `e6704edc058a` | `5aa604417` | Ready counter scaling. |
| 10 | 700 | `bnorris/pipe-static-receiver-addr` | `59c015281280` | `b8c8b1293` | Static receiver addressing and capacity protocol. |
| 11 | 740 | `bnorris/pipes-transport-emitter-refactor` | `15158bbfdfff` | `086571984` | Pipe transport emitter refactor. |
| 12 | 784 | `bnorris/pr700-3-pipe-planning` | `44feea30895f` | `88ce4f8a6` | PipeNet planning before lowering emission. |
| 13 | 780 | `bnorris/pipes-codegen-optimizations` | `51cb9909829e` | `36ddc010f`, `1cfbcb909` | PipeNet codegen cleanup and stateful one-packet write selection. |
| 14 | 687 | `bnorris/fix-683` | `3809bc553c3e` | `c10f6eab9` | Cross-block store fanout through compiler-created DFBs. |
| 15 | 680 | `bnorris/dfb-subviews-671` | `4588cf72d2fc` | `cc6f42636` | DFB block subviews and tensor-slice lowering. |
| 16 | 734 | `bnorris/pipes-multidevice-integrated-poc` | `8eebd5f9375c` | `81d1103fd` | Fabric PipeNet POC and device-domain API. Applied as feature delta `698ebd38c13e..8eebd5f9375c` because the source branch still includes older PipeNet parent state. |

Duplicate merge commits mean the source branch advanced during the refresh and
was merged again after the first merge.

## Required Inputs Missing From Current Branch

None after #734 was added in `81d1103fd`.

## Excluded Inputs

| PR | Branch | Status | Reason |
|---:|---|---|---|
| 777 | `bnorris/dfb-collection-indexing` | Closed | Excluded from this refresh. |
| 754 | `bnorris/pipes-issue-628-code-size` | Needs review | Large conflict set with the current PipeNet stack; needs separate design review before inclusion. |
| 651 | `bnorris/accumulation-scope-redesign-mainmerge` | Reference branch | Combined/reference branch only. #733 is the active extracted branch included above. |

## Conflict Notes

- #733 `python/ttl/ttl_api.py`: kept the source branch tensor recurrence
  behavior while preserving the newer runtime DFB allocation extraction.
- #783 `python/ttl/kernel_runner.py` and PipeNet docs/tests: kept computed
  receiver address metadata and the current DFB runtime configuration contract.
- #780 first merge: added an aggregate-only `PipeSendPlan::payloadTileCount`
  hook to enable one-packet writes before the source branch had the TTKernel
  cleanup design.
- #780 latest merge: removed that aggregate-only TTL lowering. The refreshed
  source implements one-packet write selection in `TTKernelCleanupPatterns.cpp`.
- #780 latest merge: kept `ttlang/Target/TargetInfo.h` for target-architecture
  helpers while preserving DFB allocation declarations from the DFB stack.
- #680 merge: added Tensor dialect registration for Python/CAPI use and split
  the fp32 running-max-subtract test input DFBs to avoid mixed unpack modes on
  one DFB.
- #734 integration: a direct merge would replay older PR700/#740 parent-stack
  state into the refreshed aggregate. The demo branch uses the feature delta
  `698ebd38c13e..8eebd5f9375c` on top of the current PipeNet planning/codegen
  stack instead.
- #734 integration: kept the current computed-address DFB runtime-argument
  allocation and added fabric runtime-argument bases after tensor, computed DFB,
  and PipeNet synchronization arguments.

## Source PR Follow-Ups

These fixes were needed while validating the aggregate branch. Port them to the
source PRs when applicable rather than leaving them only on `bnorris/demo`.

- #680 `bnorris/dfb-subviews-671`: add `tensor::TensorDialect` to
  `ttlangRegisterUpstreamDialects`, and make `ttl.ensure_dialects_registered()`
  load `ctx.dialects["tensor"]`. DFB block subviews create
  `tensor.extract_slice` directly from Python, so frontend compilation needs
  Tensor registered in a fresh MLIR context.
- #680 `bnorris/dfb-subviews-671`: update
  `test/python/test_recurrence_multi_output_dfb.py::running_max_subtract` to
  use separate input DFBs for `reduce_max` and `sub`. The fp32 validation
  rejects a single DFB feeding both default-unpack FPU consumers and
  `UnpackToDestFp32` SFPU consumers in the same compute kernel.
- #780 `bnorris/pipes-codegen-optimizations`: when rebased over the DFB stack,
  keep `kDFBAllocationsAttrName` and the operation-scoped
  `getNextAvailableDFBIndex(Operation *)` declaration in `TTL.h`, while moving
  target-architecture helpers to `ttlang/Target/TargetInfo.h`.
- #780 `bnorris/pipes-codegen-optimizations`: keep one-packet write selection in
  TTKernel cleanup and remove any stale TTL-side `PipeSendPlan::payloadTileCount`
  planning state when rebasing over older aggregate branches.
- #704 `bnorris/3-defer-intermediate-dfbs`: make
  `ConvertTTLToCompute.cpp` check replacement safety with `DominanceInfo`.
  Same-block ordering is too restrictive after compute formation when a
  replacement dominates branch-local consumers.
- #687 `bnorris/fix-683`: avoid `return` statements inside `@ttl.operation`
  functions in `control_flow_store_fanout_kernels.py`. Put the runtime
  datamovement definitions in the `_is_compile_only()` `else` branch instead.
- #733 `bnorris/4-tensor-recurrence-scopes`: update tensor recurrence tests to
  assert physical DFB allocation counts after finalize instead of checking the
  removed `ttl.compiler_allocated` marker.
- #733 `bnorris/4-tensor-recurrence-scopes`: use named integer constants for
  multi-tile slice bounds in tensor recurrence pytests and Python lit tests.
  Tuple subscripts in traced tensor-slice bounds are not resolved by the
  current frontend tracer.
- #783 `bnorris/pr700-2-computed-pipe-addresses`: update kernel-runner tests to
  use finalized `PhysicalDFBConfig` objects and tile-shape tuples after the
  logical/physical DFB configuration split.
- #687 `bnorris/fix-683`: `control_flow_store_fanout.py` Python lit coverage
  must pass TT-NN device tensors to operation calls. Mark it
  `REQUIRES: ttnn, tt-device` and use `to_dram(...)` inputs before lowering.
- #734 `bnorris/pipes-multidevice-integrated-poc`: rebase over the current
  PipeNet planning/codegen stack so the source branch can merge normally into
  `bnorris/demo`.
- #734 `bnorris/pipes-multidevice-integrated-poc`: adapt fabric lowering to the
  current `PipePlanning` and `PipeLowering` APIs. The aggregate adds explicit
  `FabricRoutePlan` and `FabricRuntimeMap` plumbing instead of using the older
  PR734 lowering entry points.
- #734 `bnorris/pipes-multidevice-integrated-poc`: compute fabric route runtime
  argument bases after tensor arguments, computed-address DFB runtime arguments,
  and PipeNet synchronization arguments.
- #734 `bnorris/pipes-multidevice-integrated-poc`: forward
  `device_domain` through both explicit and unified `@ttl.operation` decorators
  and into `_lower_program_to_kernel`.
- #734 `bnorris/pipes-multidevice-integrated-poc`: keep fabric PipeNet transfers
  on receiver-post synchronization until the capacity protocol supports
  routing-plane capacity-release atomics.
- #734 `bnorris/pipes-multidevice-integrated-poc`: keep the direct DFB helper in
  `PipeCapacityAnalysis.cpp` for `ttl.cb_pop`, and use the receiver-DFB view
  helper only for `ttl.pipe_transfer_post`.
- #734 `bnorris/pipes-multidevice-integrated-poc`: update packaging test fake
  `ttnn` modules with `SystemMeshDescriptor` and `FabricConfig`, and keep
  `FABRIC_1D` validation restricted to linear logical meshes.

## Rebuild Procedure

Enable recorded conflict reuse once:

```bash
git config rerere.enabled true
```

Refresh refs and create a dated backup before any rewrite:

```bash
git fetch origin
git switch bnorris/demo
backup_ref=backup/demo-before-refresh-$(date +%Y%m%d-%H%M%S)
git branch "$backup_ref"
git push origin "$backup_ref"
git reset --hard origin/main
git restore --source "$backup_ref" -- _demo.md
```

Merge source tips in manifest order:

```bash
git merge --no-ff --no-edit origin/bnorris/3-defer-intermediate-dfbs
git merge --no-ff --no-edit origin/bnorris/dfb-logical-physical-contract
git merge --no-ff --no-edit origin/bnorris/user-dfb-reuse
git merge --no-ff --no-edit origin/bnorris/4-tensor-recurrence-scopes
git merge --no-ff --no-edit origin/bnorris/support-indexed-pipenets
git merge --no-ff --no-edit origin/bnorris/ci-s3-publish-disk-caching
git merge --no-ff --no-edit origin/bnorris/pr700-1-pipe-schedule-validation
git merge --no-ff --no-edit origin/bnorris/pr700-2-computed-pipe-addresses
git merge --no-ff --no-edit origin/bnorris/pr700-counter-scaling
git merge --no-ff --no-edit origin/bnorris/pipe-static-receiver-addr
git merge --no-ff --no-edit origin/bnorris/pipes-transport-emitter-refactor
git merge --no-ff --no-edit origin/bnorris/pr700-3-pipe-planning
git merge --no-ff --no-edit origin/bnorris/pipes-codegen-optimizations
git merge --no-ff --no-edit origin/bnorris/fix-683
git merge --no-ff --no-edit origin/bnorris/dfb-subviews-671
```

Until #734 is rebased over the active PipeNet parent stack, apply its feature
delta after the normal source-branch merges:

```bash
git diff --binary 698ebd38c13e origin/bnorris/pipes-multidevice-integrated-poc --output=/tmp/pr734-feature-delta.patch
git apply -3 /tmp/pr734-feature-delta.patch
```

After conflicts are resolved, update this file and validate. Push the refreshed
aggregate with:

```bash
git push --force-with-lease origin bnorris/demo
```

## Validation

Completed after #734 integration:

- `python3 -m py_compile python/ttl/_src/ttl_ast.py python/ttl/kernel_runner.py python/ttl/ttl_api.py python/ttl/atom.py python/ttl/pipe.py python/sim/pipe.py test/python/test_kernel_runner.py`: passed.
- `git diff --check`: passed.
- `ninja -C build check-ttlang-mlir`: 234 passed.
- `ninja -C build check-ttlang-python-bindings`: 3 passed.
- `ninja -C build check-ttlang-packaging`: 162 passed.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird-fabric-v1.1.7 bash -lc 'source build-docker/env/activate && cmake --build build-docker --target check-ttlang-python-bindings'`: 3 passed.

Blocked after #734 integration:

- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird-fabric-v1.1.7 bash -lc 'set -o pipefail; source build-docker/env/activate && timeout 240 python -m pytest -c build-docker/test/pytest.ini --rootdir=/home/bnorris/tt/tt-lang4/test test/python/fabric/test_ping_pong.py -xvs 2>&1 | tee /tmp/device_test.log'`: failed before tt-lang kernel compilation because UMD reported `Sysmem mapped at unexpected NOC address`.
- `/home/bnorris/.local/bin/tt-smi -r all`: reset PCI device `[1]`; the UMD sysmem error persisted.
- `/usr/bin/zsh -lc 'TT_VISIBLE_DEVICES=0,1,2,3 /home/bnorris/.local/bin/tt-smi -r all'`: reset PCI devices `[0, 1, 2, 3]`; the UMD sysmem error persisted.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird-fabric-v1.1.7 bash -lc 'set -o pipefail; source build-docker/env/activate && timeout 60 python - <<PY 2>&1 | tee /tmp/device_test.log ... ttnn.get_num_devices() ... PY'`: failed with the same UMD sysmem error, confirming the runtime blocker occurs before pytest-specific code.

Completed earlier during this refresh, before #734:

- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird3-v1.1.7 bash -lc 'source build-docker/env/activate && cmake --build build-docker --target ttlang-opt'`
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird3-v1.1.7 bash -lc 'source build-docker/env/activate && cmake --build build-docker --target ttlang-opt TTLangPythonModules'`
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird3-v1.1.7 bash -lc 'source build-docker/env/activate && llvm-lit -v test/ttlang/Dialect/TTL/Transforms/insert_intermediate_dfbs.mlir test/ttlang/Dialect/TTL/Transforms/insert_intermediate_dfbs_disabled.mlir test/ttlang/Dialect/TTL/Transforms/insert_intermediate_dfbs_invalid.mlir test/ttlang/Dialect/TTL/Transforms/insert_intermediate_dfbs_nested.mlir test/ttlang/Dialect/TTL/Transforms/insert_intermediate_dfbs_control_flow.mlir test/ttlang/Dialect/TTL/Transforms/insert_intermediate_dfbs_control_flow_disabled.mlir test/ttlang/Dialect/TTL/Transforms/insert_intermediate_dfbs_control_flow_invalid.mlir'`
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird3-v1.1.7 bash -lc 'source build-docker/env/activate && llvm-lit -v test/bindings/python/ttl_autoregistration.py test/ttlang/Conversion/TTLToTTKernel/dfb_subview_store.mlir test/ttlang/Conversion/TTLToCompute/mixed_store_users.mlir test/ttlang/Conversion/TTLToCompute/bcast_lowering.mlir test/ttlang/Conversion/TTLToTTKernel/init_consolidation.mlir test/ttlang/Dialect/TTL/Transforms/convert_ttl_to_compute_multi_output.mlir'`
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird3-v1.1.7 bash -lc 'set -o pipefail; source build-docker/env/activate && timeout 180 python -m pytest -c build-docker/test/pytest.ini --rootdir=/home/bnorris/tt/tt-lang4/test test/python/test_dfb_subviews.py -xvs 2>&1 | tee /tmp/device_test.log'`
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird3-v1.1.7 bash -lc 'set -o pipefail; source build-docker/env/activate && timeout 180 python -m pytest -c build-docker/test/pytest.ini --rootdir=/home/bnorris/tt/tt-lang4/test test/python/test_tensor_slice.py::test_tensor_slice_add -xvs 2>&1 | tee /tmp/device_test.log'`
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird3-v1.1.7 bash -lc 'set -o pipefail; source build-docker/env/activate && timeout 240 python -m pytest -c build-docker/test/pytest.ini --rootdir=/home/bnorris/tt/tt-lang4/test test/python/test_recurrence_multi_output_dfb.py -xvs 2>&1 | tee /tmp/device_test.log'`
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird3-v1.1.7 bash -lc 'source build-docker/env/activate && timeout 60 python -m pytest -c build-docker/test/pytest.ini --rootdir=/home/bnorris/tt/tt-lang4/test test/python/test_layernorm.py --collect-only -q'`
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird3-v1.1.7 bash -lc 'source build-docker/env/activate && timeout 120 python -m pytest -c build-docker/test/pytest.ini --rootdir=/home/bnorris/tt/tt-lang4/test test/packaging/test_workflow_helper_scripts.py -q'`
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird3-v1.1.7 bash -lc 'source build-docker/env/activate && llvm-lit -v test/ttlang/Dialect/TTL/Transforms/pipe_stateful_one_packet_write.mlir test/ttlang/Dialect/TTL/Transforms/pipe_loop_invariant_cleanup.mlir test/ttlang/Dialect/TTL/Transforms/convert_pipe_completion_resources.mlir test/ttlang/Dialect/TTL/Transforms/convert_pipe_ops.mlir test/ttlang/Dialect/TTL/Transforms/convert_pipe_ops_overlap.mlir test/ttlang/Dialect/TTL/Transforms/pipe_published_address_overlap.mlir'`
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird3-v1.1.7 bash -lc 'source build-docker/env/activate && llvm-lit -v test/ttlang/Dialect/TTL/Transforms/insert_intermediate_dfbs_deferred.mlir'`
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird3-v1.1.7 bash -lc 'source build-docker/env/activate && timeout 60 python -m pytest -c build-docker/test/pytest.ini --rootdir=/home/bnorris/tt/tt-lang4/test test/python/test_control_flow_store_fanout.py --collect-only -q'`: collected 12 tests.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird3-v1.1.7 bash -lc 'set -o pipefail; source build-docker/env/activate && timeout 240 python -m pytest -c build-docker/test/pytest.ini --rootdir=/home/bnorris/tt/tt-lang4/test test/python/test_control_flow_store_fanout.py -xvs 2>&1 | tee /tmp/device_test.log'`: 12 passed.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird3-v1.1.7 bash -lc 'set -o pipefail; source build-docker/env/activate && timeout 180 python -m pytest -c build-docker/test/pytest.ini --rootdir=/home/bnorris/tt/tt-lang4/test test/python/test_kernel_runner.py -xvs 2>&1 | tee /tmp/device_test.log'`: 32 passed.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird3-v1.1.7 bash -lc 'set -o pipefail; source build-docker/env/activate && timeout 420 python -m pytest -c build-docker/test/pytest.ini --rootdir=/home/bnorris/tt/tt-lang4/test test/python/test_tensor_recurrences.py -xvs 2>&1 | tee /tmp/device_test.log'`: 38 passed, 2 xfailed.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird3-v1.1.7 bash -lc 'source build-docker/env/activate && llvm-lit -v build-docker/test --filter="tensor_recurrence_dst_acc.py|control_flow_store_fanout.py"'`: 2 passed.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird3-v1.1.7 bash -lc 'set -o pipefail; source build-docker/env/activate && timeout 3600 ninja -C build-docker check-ttlang-all 2>&1 | tee /tmp/device_test.log'`: passed.

Full `check-ttlang-all` results:

- `check-ttlang-mlir`: 230 passed.
- `check-ttlang-python-bindings`: 3 passed.
- `check-ttlang-packaging`: 162 passed.
- `check-ttlang-pytest`: 1982 passed, 3 skipped, 8 xfailed.
- `check-ttlang-python-lit`: 86 passed, 1 unsupported.
- `check-ttlang-me2e`: 868 passed, 35 xfailed.
