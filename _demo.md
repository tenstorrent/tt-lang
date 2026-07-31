# Demo Integration Branch

Branch: `bnorris/demo`.

Base: `origin/main` at `711eddc4a9aa`
(`[ci] re-enable Galaxy hardware tests (#789)`).

Purpose: disposable aggregate branch for integration testing and the
all-gather minimal matmul examples. This branch is not a source PR parent.

## Update Policy

- Maintain source PR branches independently using the workflow in
  `/home/bnorris/tt/doc/StackedPRs.md`.
- Fetch remote refs and create a dated, pushed backup branch before changing
  `bnorris/demo`.
- Merge live source branch tips with `git merge --no-ff`. Do not use
  cherry-picks as the normal source-branch update mechanism.
- Restore demo-only checkpoints as explicit `[demo baseline]` or
  `[demo item N]` commits after the source aggregate is current.
- Record every aggregate-only repair below and apply the corresponding fix to
  its source PR when required.
- Validate the refreshed baseline on Galaxy before checklist work. Validate
  every checklist item on Galaxy before starting the next item.

## Current Backups

| Backup | Commit | Purpose |
|---|---|---|
| `backup/demo-before-latest-refresh-20260730-132242` | `c7a8d94ecc37` | Last four-device-validated demo and item 1 checkpoint. |
| `backup/demo-before-pipes-codegen-refresh-20260730-145611` | `1ce92bc1` | Aggregate before the latest PR780 restack merge. |
| `backup/demo-before-checklist-20260730-130603` | recorded remote ref | Pre-checklist demo checkpoint. |

## Included Source Tips

The following refs were fetched and verified as ancestors of `bnorris/demo` on
2026-07-30. A source branch that advances requires another dated backup,
merge, and validation cycle.

| Order | PR | Source branch | Included tip |
|---:|---:|---|---|
| 1 | 704 | `bnorris/3-defer-intermediate-dfbs` | `d102e7d16828` |
| 2 | 778 | `bnorris/dfb-logical-physical-contract` | `e284c90ff20c` |
| 3 | 775 | `bnorris/user-dfb-reuse` | `b92ef2da0fdd` |
| 4 | 733 | `bnorris/4-tensor-recurrence-scopes` | `ff2cc828b4f1` |
| 5 | 673 | `bnorris/support-indexed-pipenets` | `713892fa81df` |
| 6 | 782 | `bnorris/pr700-1-pipe-schedule-validation` | `fd89d1aa9117` |
| 7 | 783 | `bnorris/pr700-2-computed-pipe-addresses` | `2b226bd030d2` |
| 8 | 765 | `bnorris/pr700-counter-scaling` | `1dd43c6ba035` |
| 9 | 700 | `bnorris/pipe-static-receiver-addr` | `03b58ae7f6fd` |
| 10 | 740 | `bnorris/pipes-transport-emitter-refactor` | `cbd883bd8804` |
| 11 | 784 | `bnorris/pr700-3-pipe-planning` | `efd39395e8ec` |
| 12 | 780 | `bnorris/pipes-codegen-optimizations` | `fe705d8cae41` |
| 13 | 687 | `bnorris/fix-683` | `a4ca54878a7b` |
| 14 | 680 | `bnorris/dfb-subviews-671` | `4588cf72d2fc` |
| 15 | 734 | `bnorris/pipes-multidevice-integrated-poc` | `14c5621d299e` |
| 16 | 754 | `bnorris/pipes-issue-628-code-size` | `8501c45aebe8` |
| 17 | 795 | `bnorris/pr734-operation-device-domain` | `2dc25d5487d0` |

PR780 was force-pushed from the previously integrated history to the current
restack. Merge `34745510` records `fe705d8cae41` as an explicit parent, so
future ancestry checks identify the current source tip without replaying
equivalent patches.

## Integration Checkpoints

| Commit | Description |
|---|---|
| `34745510` | Records the latest PR780 restack as an explicit merge parent. |
| `78eea62f` | Restores selected PipeNet graph state, immutable plans, fabric route dependencies, and PR687 control-flow helpers lost during aggregate conflict resolution. |
| `1d0ff827` | Merges PR704 at `d102e7d1` and adapts PR687 compute-result replacement to the deferred materialization API. |
| `d603c90e` | Merges PR754 at `8501c45a`, including mixed local/global selected counters and current DFB/selected-op test contracts. |
| `58d6c82f` | Restores bounded per-edge lowering for graphs with at most 12 records; larger graphs use compact selected-pipe lowering. |
| `a3021dca` | Restores the baseline direct and context-manager all-gather matmul examples. |
| `37c570f5` | Restores checklist item 1 row-broadcast bias variants. |
| `339b9b69` | Restores Tensor dialect C API registration required by PR680 DFB subviews. |
| `4d71475d` | Merges PR795 operation device-domain forwarding into the aggregate. |

## Aggregate Repairs And Source Follow-Ups

These changes must not remain undocumented aggregate behavior.

- PR687 `bnorris/fix-683`: replace producer-compute uses independently when
  the replacement dominates each user. The previous all-or-nothing
  `replaceOpIfSafe` decision left nested DFB consumers attached to the tensor
  operation after PR704 deferred materialization. The aggregate uses
  `DominanceInfo` and erases the tensor operation only when no uses remain.
- PR704 `bnorris/3-defer-intermediate-dfbs`: retain PR687
  `addMaterializationUse` handling when adapting materialization lookup from a
  direct result to `std::optional<OpResult>`.
- PR754 `bnorris/pipes-issue-628-code-size`: rebase over the current selected
  PipeNet and DFB contracts. Handwritten selected-pipe tests require the
  `net`, device-index, and collective fields; verifier tests require finalized
  user DFB IDs before `ttl-verify-pipenet-guards`.
- PR754: preserve `index` receiver counts for selected-record predicates and
  cast once to `i32` for NoC multicast payload and atomic operands.
- PR754: preserve mixed local/global selected counter allocation and typed
  `ttkernel.cast_to_l1_addr` selection. Allocating all selected counters
  globally is unnecessary and increases runtime arguments.
- PR734 `bnorris/pipes-multidevice-integrated-poc`: keep fabric route runtime
  arguments after tensor, computed-address DFB, and PipeNet synchronization
  arguments. Keep selected fabric transfers on receiver-post synchronization
  until capacity-release atomics are implemented for routing-plane transport.
- PR795 `bnorris/pr734-operation-device-domain`: keep `device_domain` on the
  public `ttl.operation` dispatcher and forward it through explicit and unified
  operation lowering. This PR targets PR734 so the fabric frontend remains
  valid independently of the demo aggregate.
- PR680 `bnorris/dfb-subviews-671`: retain Tensor dialect registration for
  Python-created `tensor.extract_slice` operations. Keep fp32 recurrence tests
  from sharing one input DFB across incompatible unpack modes.
- PR780 `bnorris/pipes-codegen-optimizations`: preserve DFB allocation
  declarations while using `ttlang/Target/TargetInfo.h`. Keep one-packet write
  selection in TTKernel cleanup without stale TTL-side payload-count planning.

The 12-record frontend threshold in `58d6c82f` is demo-specific. It preserves
the four-device all-to-all implementation that was previously hardware
validated while allowing larger graphs to compile with compact selected-pipe
lowering. It is not a substitute for validating the compact runtime protocol.

## Demo Checklist

The runnable commands and correctness method are documented in
`examples/all_gather_minimal_matmul/README.md`.

| Item | Implementation | Refreshed Galaxy validation |
|---|---|---|
| Baseline: K-sharded activation all-gather and N-sharded matmul | Restored at `a3021dca` | Pending |
| 1: N-sharded row-broadcast bias | Restored at `37c570f5` | Pending |
| 2: full-grid multi-node scheduling | Saved in `stash@{2026-07-30 13:08:27}`; not applied | Not started |
| 3a: ReLU | Not started | Not started |
| 3b: GELU and SiLU | Not started | Not started |
| 4: chunked N output | Not started | Not started |
| 5: addcmul | Not started | Not started |
| 6: SwiGLU | Not started | Not started |
| 7: transpose selection | Not started | Not started |
| 8: fabric worker configuration | Not started | Not started |
| 9: FSDP weight gather | Not started | Not started |

The prior `c7a8d94e` snapshot passed direct and context-manager baseline and
item 1 execution on four devices. The restored example files are identical,
but the compiler aggregate changed; those results are reference evidence, not
validation of the refreshed commits.

## Current Validation

Completed for source aggregate `d603c90e`:

- `cmake --build build`: passed.
- `ninja -C build check-ttlang-mlir`: 241 passed.
- Repaired MLIR tests were run individually before the complete MLIR suite.

Completed after restoring the demo checkpoints:

- `python3 -m py_compile` for baseline and item 1 modules and launchers:
  passed.
- Restored files match the recorded `9a59766a` baseline and `c7a8d94e` item 1
  snapshots byte-for-byte before this manifest update.
- `test/bindings/python/ttl_autoregistration.py`: passed.
- `test/python/test_dfb_runtime_config.py`: 19 passed.
- `test/python/test_dfb_subviews.py` and selected collective and mixed-counter
  PipeNet tests: 14 passed.
- `test/python/test_ttl_api_device_options.py`: 17 passed.
- Docker rebuild in `bnorris-ird-fabric-v1.1.7`: passed.
- Four-device baseline startup did not reach kernel compilation. Fabric router
  synchronization timed out on device 2; after reset, topology discovery
  exposed only a 2x1 mesh. Local four-device validation remains pending until
  all four devices form a connected fabric mesh.

Pending before checklist item 2:

- Run baseline direct and context-manager examples on four local devices.
- Run item 1 direct and context-manager examples on four local devices.
- Run baseline direct and context-manager examples on Galaxy.
- Run item 1 direct and context-manager examples on Galaxy.
- Record exact commands, result metrics, device count, container, and commit.

## Refresh Procedure

```bash
git fetch origin
git switch bnorris/demo
backup_ref=backup/demo-before-refresh-$(date +%Y%m%d-%H%M%S)
git branch "$backup_ref"
git push origin "$backup_ref"
```

Merge changed source refs in table order. If source branches were force-pushed,
rebuild from `origin/main` and merge the live tips instead of replaying obsolete
merge commits. Restore validated demo-only checkpoints after the source
aggregate is complete.

Before publishing:

```bash
git diff --check
ninja -C build check-ttlang-mlir
git push --force-with-lease origin bnorris/demo
```
