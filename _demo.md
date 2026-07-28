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

Not run:

- Device pytests for the aggregate. They require the Docker hardware-test protocol.
