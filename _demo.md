# Demo Integration Branch

Branch: `bnorris/demo`.

Base: `origin/main` at `d6aec0dc26b9` (`[test] Isolate build-specific test configuration (#774)`).

Purpose: disposable aggregate branch for demos and integration testing. This branch is not a PR review layer and should not become a parent for source PRs.

## Policy

- Maintain the underlying review stacks with Graphite.
- Maintain `bnorris/demo` with explicit git merge commits from the refreshed source tips.
- Rebuild `bnorris/demo` from `origin/main` when source branches rebase or force-push.
- Do not use cherry-picks as the default update mechanism. If a short-lived cherry-pick is used for an emergency demo repair, record it in this file and replace it with a source-branch merge on the next rebuild.
- Keep this file updated with the base SHA, source tips, included branches, excluded branches, conflict notes, and validation.

## Source Of Truth

- Current refresh status: `/home/bnorris/tt/tt-lang-acc-3-defer-intermediate-dfbs/_updates.md`.
- PR inventory: `/home/bnorris/tt/PRs.md`. It is useful context, but some sections are stale.
- Refreshed local refs in this checkout use `refs/remotes/prrefresh/bnorris/...`.

## Included Inputs

| Order | PR | Source branch | Tip | Merge commit | Notes |
|---:|---:|---|---|---|---|
| 1 | 704 | `bnorris/3-defer-intermediate-dfbs` | `25fab627fec9` | `74ba938b1` | Deferred compiler-created intermediate DFB materialization. |
| 2 | 777 | `bnorris/dfb-collection-indexing` | `2c1846266308` | `70dd9249b` | Statically indexed frontend DFB collections. |
| 3 | 778 | `bnorris/dfb-logical-physical-contract` | `c86d612d3499` | `a46efa5ae` | Logical DFB identity and physical-allocation contract. |
| 4 | 775 | `bnorris/user-dfb-reuse` | `420f03dc74fd` | `5d3d1ab2e` | Concurrent lifetime analysis and physical DFB index reuse. |
| 5 | 733 | `bnorris/4-tensor-recurrence-scopes` | `80c0c07d0a5d` | `0944cdc8f` | Tensor recurrence lowering through DST accumulation scopes. |
| 6 | 773 | `bnorris/consolidate-wheel-builds` | `63fa59255d1d` | `c374bd033` | Consolidated manylinux wheel build workflow. |
| 7 | 673 | `bnorris/support-indexed-pipenets` | `aac6a1b0ff9e` | `d67b16453` | Indexed PipeNet receiver expressions. |
| 8 | 759 | `bnorris/ci-s3-publish-disk-caching` | `24af3db2b4a8` | `7063cb21c` | S3 publish disk caching and wheel workflow fixes. |
| 9 | 700 | `bnorris/pipe-static-receiver-addr` | `b2a98d5a459e` | `bd337b517` | Static PipeNet receiver addressing. |
| 10 | 740 | `bnorris/pipes-transport-emitter-refactor` | `205c5de3959e` | `886afc01e` | Pipe transport emitter refactor. |
| 11 | 734 | `bnorris/pipes-multidevice-integrated-poc` | `b8268c7415f6` | `45f7bf4df` | Multidevice PipeNet integrated POC. |
| 12 | 687 | `bnorris/fix-683` | `6ccb13241c23` | `90b38645b` | Cross-block store fanout through compiler DFBs. |
| 13 | 680 | `bnorris/dfb-subviews-671-mainmerge` | `991a01c71148` | `a9b91c5be` | DFB block subviews and re-consumed output ordering fix. |

## Excluded Inputs

| PR | Branch | Tip | Reason |
|---:|---|---|---|
| 765 | `bnorris/pr700-counter-scaling` | `782bdd338f34` | Excluded after merge attempt. It conflicts with the already-included multidevice PipeNet POC in `ConvertTTLToTTKernel.cpp`, `PipeLowering.cpp`, `PipeLowering.h`, and `python/ttl/kernel_runner.py`. The conflict is over the PipeNet counter/resource model: #765 scales PR700 local synchronization counters, while #734 introduces multidevice PipeNet lowering, fabric/global semaphore storage, and a broader resource plan. |
| 651 | `bnorris/accumulation-scope-redesign-mainmerge` | `dea27f8d2a01` | Excluded after merge attempt. `PRs.md` marks #651 as the combined development branch retained for reference and states that it will not merge; #733 is the extracted active branch from that stack and is already included. The staging branch also conflicts across the accumulation pass stack, DFB materialization, compute lowering, pipeline ordering, docs, and tests. |
| 754 | `bnorris/pipes-issue-628-code-size` | `f5e5743e2209` | Excluded from the initial input set. The parent merge produced 251 conflict hunks across 34 files; it needs a separate design review before inclusion. |

Older PRs in `/home/bnorris/tt/PRs.md` are not part of this demo input set until refreshed and added here deliberately.

## Conflict Notes

- #778 `python/ttl/atom.py`: kept current lifted DFB config handling with the branch's logical/physical contract.
- #759 workflow/scripts: preserved wheel consolidation, included S3 publish disk caching, and kept updated hardware job result checks.
- #700 kernel runner: preserved current `PhysicalDFBConfig`, tuple compatibility, computed-address backing tensor budget exclusion, and tile descriptors.
- #734 Python frontend/sim/kernel runner tests: kept indexed PipeNets, multidevice `DeviceDomain`, static PipeNet metadata discovery, graph/device-domain helpers, and capture handling.
- #687 `TTLInsertIntermediateDFBs.cpp`: integrated cross-block store fanout into the current deferred materialization pass.
- #680 `ConvertTTLToCompute.cpp`: restored relocation of absorbed-store releases and same-DFB consumers so a re-consumed output keeps `ttl.cb_push` before `ttl.cb_wait`.

## Source PR Follow-Ups

These fixes were made while validating the aggregate branch. They should be ported back to the source branches instead of remaining only on `bnorris/demo`.

- #680 `bnorris/dfb-subviews-671`: update `replacementDominatesRemainingUses` in `lib/Dialect/TTL/Transforms/ConvertTTLToCompute.cpp` to use `DominanceInfo` instead of requiring every remaining use to be in the same block as the replacement compute. The same-block check rejects valid users inside nested control-flow regions that are dominated by the compute, leaving tensor-level DFB materialization without the expected producer-side push/pop sequence.
- #680 `bnorris/dfb-subviews-671`: add `tensor::TensorDialect` to `ttlangRegisterUpstreamDialects`, and make `ttl.ensure_dialects_registered()` load `ctx.dialects["tensor"]`. DFB block subviews create `tensor.extract_slice` directly from Python, so the Python MLIR context must register Tensor before frontend compilation.
- #680 `bnorris/dfb-subviews-671`: update `test/python/test_recurrence_multi_output_dfb.py::running_max_subtract` to use separate input DFBs for `reduce_max` and `sub`. Current fp32 unpack-mode validation rejects one CB feeding both default-unpack FPU consumers and `UnpackToDestFp32` SFPU consumers in the same compute kernel.
- #733 `bnorris/4-tensor-recurrence-scopes`: update `test/python/test_tensor_recurrences.py` so the post-optimization assertion in `test_resident_contribution_early_pop_does_not_form_dst` accounts for finalized DFB allocation metadata, and so multi-tile tensor slices use integer globals instead of `MULTI_TILE_SHAPE[...]` tuple subscripts in DSL slice bounds. After `ttl-finalize-dfb-indices` and TTL-to-TTKernel lowering, `ttl.compiler_allocated` no longer appears in the final MLIR; the test should check the extra finalized allocation instead.
- #700 `bnorris/pipe-static-receiver-addr`: when this branch is refreshed on top of the DFB identity/physical allocation stack, update the handwritten PipeNet verifier tests to include current DFB metadata. In `verify_pipenet_guards.mlir`, add `ttl.dfb_allocations = []` to the final two split modules and add explicit `dfb_id` attributes to their `ttl.bind_cb` ops. In `verify_pipenet_schedule_invalid.mlir`, add explicit `dfb_id` attributes to all user-declared `ttl.bind_cb` ops that currently omit them.
- #734 `bnorris/pipes-multidevice-integrated-poc`: update `test/ttlang_test_utils.py` to validate `FABRIC_1D` mesh requests before opening the mesh, and update `test/packaging/test_ttlang_test_utils.py` mocks for the current `ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()` API.
- #734 `bnorris/pipes-multidevice-integrated-poc`: update `test/python/test_kernel_runner.py::test_emit_runner_source_uses_shared_pipe_resource_helpers` to use a concrete reader config descriptor for its NOC `KernelSpec`. Current `emit_runner_source()` serializes NOC roles from `ReaderConfigDescriptor` and `WriterConfigDescriptor`; a placeholder `object()` is no longer a valid NOC kernel config.
- #759 `bnorris/ci-s3-publish-disk-caching`: when refreshed on top of the shared manylinux workflow cleanup, keep `publish-s3-pypi.yml` free of multiline `run: |` steps. Express selected-wheel build success requirements in the `publish` job condition rather than a multiline shell verification step.

## Local Main Hygiene Fixes

- Issue #785: add `tensor::TensorDialect` to `TTLConvertTTLToTTKernel` `dependentDialects`. The pass creates and legalizes Tensor dialect ops, so it must register Tensor explicitly when loaded into a context that has not already loaded the dialect.

## Rebuild Procedure

Enable recorded conflict reuse once:

```bash
git config rerere.enabled true
```

Import refreshed local refs:

```bash
git fetch /home/bnorris/tt/tt-lang-acc-3-defer-intermediate-dfbs \
  bnorris/3-defer-intermediate-dfbs:refs/remotes/prrefresh/bnorris/3-defer-intermediate-dfbs \
  bnorris/dfb-collection-indexing:refs/remotes/prrefresh/bnorris/dfb-collection-indexing \
  bnorris/dfb-logical-physical-contract:refs/remotes/prrefresh/bnorris/dfb-logical-physical-contract \
  bnorris/user-dfb-reuse:refs/remotes/prrefresh/bnorris/user-dfb-reuse \
  bnorris/4-tensor-recurrence-scopes:refs/remotes/prrefresh/bnorris/4-tensor-recurrence-scopes \
  bnorris/consolidate-wheel-builds:refs/remotes/prrefresh/bnorris/consolidate-wheel-builds \
  bnorris/support-indexed-pipenets:refs/remotes/prrefresh/bnorris/support-indexed-pipenets \
  bnorris/ci-s3-publish-disk-caching:refs/remotes/prrefresh/bnorris/ci-s3-publish-disk-caching \
  bnorris/pipe-static-receiver-addr:refs/remotes/prrefresh/bnorris/pipe-static-receiver-addr \
  bnorris/pipes-transport-emitter-refactor:refs/remotes/prrefresh/bnorris/pipes-transport-emitter-refactor \
  bnorris/pipes-multidevice-integrated-poc:refs/remotes/prrefresh/bnorris/pipes-multidevice-integrated-poc \
  bnorris/fix-683:refs/remotes/prrefresh/bnorris/fix-683 \
  bnorris/dfb-subviews-671-mainmerge:refs/remotes/prrefresh/bnorris/dfb-subviews-671-mainmerge
```

Rebuild the aggregate from current `origin/main`:

```bash
git fetch origin
git switch bnorris/demo
backup_ref=backup/demo-before-refresh-$(date +%Y%m%d-%H%M%S)
git branch "$backup_ref"
git reset --hard origin/main
git restore --source "$backup_ref" -- _demo.md

git merge --no-ff --no-edit refs/remotes/prrefresh/bnorris/3-defer-intermediate-dfbs
git merge --no-ff --no-edit refs/remotes/prrefresh/bnorris/dfb-collection-indexing
git merge --no-ff --no-edit refs/remotes/prrefresh/bnorris/dfb-logical-physical-contract
git merge --no-ff --no-edit refs/remotes/prrefresh/bnorris/user-dfb-reuse
git merge --no-ff --no-edit refs/remotes/prrefresh/bnorris/4-tensor-recurrence-scopes
git merge --no-ff --no-edit refs/remotes/prrefresh/bnorris/consolidate-wheel-builds
git merge --no-ff --no-edit refs/remotes/prrefresh/bnorris/support-indexed-pipenets
git merge --no-ff --no-edit refs/remotes/prrefresh/bnorris/ci-s3-publish-disk-caching
git merge --no-ff --no-edit refs/remotes/prrefresh/bnorris/pipe-static-receiver-addr
git merge --no-ff --no-edit refs/remotes/prrefresh/bnorris/pipes-transport-emitter-refactor
git merge --no-ff --no-edit refs/remotes/prrefresh/bnorris/pipes-multidevice-integrated-poc
git merge --no-ff --no-edit refs/remotes/prrefresh/bnorris/fix-683
git merge --no-ff --no-edit refs/remotes/prrefresh/bnorris/dfb-subviews-671-mainmerge
```

After conflicts are resolved, update this file with the new base SHA, source tips, conflict notes, exclusions, and validation. Commit the file on `bnorris/demo`.

## Validation

Completed while creating this branch:

- `cmake --build build --target ttlang-opt`.
- `cmake --build build --target TTLangPythonModules`.
- `env PYTHONPYCACHEPREFIX=/tmp/tt-lang4-pyc python3 -m py_compile python/ttl/_src/ttl_ast.py test/python/test_recurrence_multi_output_dfb.py`.
- `/home/bnorris/.local/bin/pre-commit run clang-format --files lib/Dialect/TTKernel/Transforms/TTKernelInsertInits.cpp lib/Dialect/TTL/Transforms/ConvertTTLToCompute.cpp`.
- `llvm-lit test/ttlang/Dialect/TTL/Transforms/insert_intermediate_dfbs_control_flow.mlir test/ttlang/Dialect/TTL/Transforms/insert_intermediate_dfbs_control_flow_disabled.mlir test/ttlang/Dialect/TTL/Transforms/insert_intermediate_dfbs_control_flow_invalid.mlir test/ttlang/Dialect/TTL/Transforms/insert_intermediate_dfbs_disabled.mlir test/ttlang/Dialect/TTL/Transforms/insert_intermediate_dfbs_deferred.mlir`.
- `llvm-lit test/ttlang/Conversion/TTLToCompute/bcast_lowering.mlir test/ttlang/Conversion/TTLToCompute/matmul_fusion.mlir test/ttlang/Conversion/TTLToCompute/mixed_store_users.mlir test/ttlang/Conversion/TTLToTTKernel/dfb_subview_store.mlir test/ttlang/Conversion/TTLToTTKernel/init_consolidation.mlir test/ttlang/Dialect/TTL/Transforms/convert_ttl_to_compute_multi_output.mlir`.

Completed after the first Docker `check-ttlang-all` attempt:

- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird1-v1.1.7 bash -c 'source build-docker/env/activate && cmake --build build-docker --target TTLangPythonModules'`.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird1-v1.1.7 bash -c 'source build-docker/env/activate && llvm-lit -vv test/ttlang/Dialect/TTL/Transforms/verify_pipenet_guards.mlir'`.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird1-v1.1.7 bash -c 'source build-docker/env/activate && llvm-lit -vv test/ttlang/Dialect/TTL/Transforms/verify_pipenet_schedule_invalid.mlir'`.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird1-v1.1.7 bash -c 'source build-docker/env/activate && cmake --build build-docker --target ttlang-opt'`.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird1-v1.1.7 bash -c 'set -o pipefail; source build-docker/env/activate && timeout 600 ninja -C build-docker check-ttlang-python-bindings 2>&1 | tee /tmp/device_test.log'` passed with `3 passed`.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird1-v1.1.7 bash -c 'source build-docker/env/activate && llvm-lit -vv test/ttlang/Dialect/TTL/Transforms/insert_intermediate_dfbs_deferred.mlir'`.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird1-v1.1.7 bash -c 'source build-docker/env/activate && ninja -C build-docker check-ttlang-mlir'`.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird1-v1.1.7 bash -c 'source build-docker/env/activate && python -m pytest --rootdir=/home/bnorris/tt/tt-lang4 -c /dev/null -q test/packaging/test_ttlang_test_utils.py::test_fabric_1d_uses_linear_logical_mesh test/packaging/test_ttlang_test_utils.py::test_fabric_1d_rejects_non_linear_logical_mesh test/packaging/test_workflow_helper_scripts.py::test_s3_publish_requires_every_selected_wheel_build_to_succeed test/packaging/test_workflow_helper_scripts.py::test_shared_manylinux_workflows_have_no_multiline_shell'`.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird1-v1.1.7 bash -c 'source build-docker/env/activate && ninja -C build-docker check-ttlang-packaging'`.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird1-v1.1.7 bash -c 'set -o pipefail; source build-docker/env/activate && timeout 180 python -m pytest test/python/test_dfb_subviews.py::test_dfb_subview_pack_unpack -xvs 2>&1 | tee /tmp/device_test.log'`.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird1-v1.1.7 bash -c 'set -o pipefail; source build-docker/env/activate && timeout 180 python -m pytest test/python/test_dfb_subviews.py -xvs 2>&1 | tee /tmp/device_test.log'`.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird1-v1.1.7 bash -c 'set -o pipefail; source build-docker/env/activate && timeout 180 python -m pytest test/python/test_recurrence_multi_output_dfb.py::test_running_max_subtract[fp32] -xvs 2>&1 | tee /tmp/device_test.log'`.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird1-v1.1.7 bash -c 'set -o pipefail; source build-docker/env/activate && timeout 300 python -m pytest test/python/test_tensor_recurrences.py::test_resident_contribution_early_pop_does_not_form_dst test/python/test_tensor_recurrences.py::test_multi_tile_block_recurrence test/python/test_tensor_recurrences.py::test_multi_tile_distinct_per_iteration_contributions test/python/test_tensor_recurrences.py::test_three_accumulators_multi_tile_block -xvs 2>&1 | tee /tmp/device_test.log'`.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird1-v1.1.7 bash -c 'set -o pipefail; source build-docker/env/activate && TTLANG_INITIAL_MLIR=/tmp/tensor_recurrence_dst_acc.initial.mlir timeout 300 python test/python/tensor_recurrence_dst_acc.py 2>&1 | tee /tmp/device_test.log'`.
- `docker exec -w /home/bnorris/tt/tt-lang4/test/python bnorris-ird1-v1.1.7 bash -c 'set -o pipefail; source /home/bnorris/tt/tt-lang4/build-docker/env/activate && timeout 1800 python -m pytest -c /home/bnorris/tt/tt-lang4/build-docker/test/pytest.ini --rootdir=/home/bnorris/tt/tt-lang4/test /home/bnorris/tt/tt-lang4/test/python -v --tb=short -x 2>&1 | tee /tmp/device_test.log'` passed with `2034 passed, 17 skipped, 8 xfailed`.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird1-v1.1.7 bash -c 'set -o pipefail; source build-docker/env/activate && timeout 1800 ninja -C build-docker check-ttlang-pytest 2>&1 | tee /tmp/device_test.log'` passed with `2034 passed, 17 skipped, 8 xfailed`.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird1-v1.1.7 bash -c 'set -o pipefail; source build-docker/env/activate && timeout 1800 ninja -C build-docker check-ttlang-python-lit 2>&1 | tee /tmp/device_test.log'` passed with `86 passed, 1 unsupported`.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird1-v1.1.7 bash -c 'set -o pipefail; source build-docker/env/activate && timeout 1800 ninja -C build-docker check-ttlang-me2e 2>&1 | tee /tmp/device_test.log'` passed with `868 passed, 35 xfailed`.
- `docker exec -w /home/bnorris/tt/tt-lang4 bnorris-ird1-v1.1.7 bash -c 'set -o pipefail; source build-docker/env/activate && timeout 3600 ninja -C build-docker check-ttlang-all 2>&1 | tee /tmp/device_test.log'` passed. Subtargets reported: `check-ttlang-mlir` `230/230`, `check-ttlang-python-bindings` `3 passed`, `check-ttlang-packaging` `162 passed`, `check-ttlang-pytest` `2034 passed, 17 skipped, 8 xfailed`, `check-ttlang-python-lit` `86 passed, 1 unsupported`, and `check-ttlang-me2e` `868 passed, 35 xfailed`.
