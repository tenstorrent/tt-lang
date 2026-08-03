# Demo Integration Branch

Target branch: `bnorris/demo`.

Current rebuild branch: `integration/demo-rebuild-20260802-v4`.

Base: `origin/main` at `711fcb38`.

Purpose: integrate current unmerged compiler work for validation and the
all-gather minimal matmul examples. This branch is not a source PR parent.

## Update Policy

- Maintain source PR branches independently using
  `/home/bnorris/tt/doc/StackedPRs.md`.
- Fetch source refs and create a dated, pushed backup before rebuilding.
- Rebuild from `origin/main` when constituent branches have been rewritten.
- Integrate reviewed PR deltas only. Do not apply an aggregate development
  branch as one tree delta when it includes commits outside the reviewed PRs.
- Restore demo-only commits only after all source branches are integrated.
- Record every integration repair below, including its source owner and the
  required source-branch action.
- Run Docker `check-ttlang-all` after a major rebuild and before Galaxy tests.
- Validate each demo checklist item on Galaxy before starting the next item.

## Current Rebuild

Started: 2026-08-02.

Backups:

- `backup/demo-before-rebuild-20260802-192524` at `db9d2dfb` (pushed).
- `backup/demo-rebuild-v2-before-main-refresh-20260802-205901` at `ed897a86`
  (pushed).
- `backup/demo-rebuild-v3-before-current-stack-20260802-212557` at `5041479b`
  (pushed).

The prior accumulation aggregate `fba66fa8` was not applied as one delta. It
includes development commits outside the reviewed PR heads. The current
rebuild applies each reviewed final delta to merged PR704 instead.

| Order | PR or branch | Source tip | Integration commit | Status |
|---:|---|---|---|---|
| 1 | `origin/main`, including #704 | `711fcb38` | - | Applied |
| 2 | #733 tensor recurrence scopes | `7e743a06` | `419c6a8a` | Applied |
| 3 | #778 logical DFB identities | `c9c4a0a5` | `b4cae829` | Applied |
| 4 | #775 user DFB reuse | `94a42bb1` | `c312d47c` | Applied |
| 5 | #673 indexed PipeNet receivers | `713892fa` | `bffe5b31` | Applied |
| 6 | #687 control-flow stores | `a4ca5487` | `973fe822`, `ba690d18` | Applied and repaired |
| 7 | #680 DFB subviews | `4588cf72` | `3c4bd203`, `eddbfea9` | Applied and repaired |
| 8 | #782-#784 PipeNet stack | `fc9233b1`, `e4bcf154`, `4d003e09`, `bd3a991e`, `671ca189`, `9213a58b` | `684d73ae` | Applied and repaired |
| 9 | #780 grouped PipeTransport | `374909df` | Pending | Pending |
| 10 | #734 multidevice fabric | Pending refresh | Pending | Pending |
| 11 | #754 compact selected PipeNets | Pending refresh | Pending | Pending |
| 12 | #795 operation device domains | Pending refresh | Pending | Pending |
| 13 | demo examples | `a3021dca`, `37c570f5` | Pending | Pending |

## Integration Repairs And Source Follow-Ups

Each repair must be applied to the named source PR or remain identified as a
composition change between independent PRs.

- #775 with #733: preserve recurrence lowering and add
  `reuse_user_dfbs_flag` to DFB finalization. Owner: composition between #733
  and #775; no source PR change is required.
- #687 multi-block stores: evaluate mutual exclusion against every direct
  store, including stores in the producer block. Continue using only
  out-of-block stores as clone targets. The source implementation evaluates a
  single branch store in isolation and contradicts its own materialization
  tests. Owner: #687 when refreshed onto merged #704.
- #687 with #704: implement multi-block store handling in immutable
  `IntermediateDFBPlanning`; do not restore mutation during analysis. Owner:
  #687 when refreshed onto merged #704.
- #680 direct block broadcast: record the broadcast input in
  `ComputeOpCreationPlan`. The source planner otherwise emits one input map
  with zero compute inputs. Owner: #680 when refreshed onto merged #704.
- #680 computed block broadcast: dispatch an unattached broadcast input to the
  existing planned fusion mechanism. In-tile broadcast uses an intermediate
  DFB; inter-tile broadcast fuses with its producer. Owner: #680 when refreshed
  onto merged #704.
- #680 conversion test: run producer creation and intermediate DFB insertion
  before final conversion. Final conversion must diagnose unassigned stores;
  it must not leave deliberately invalid IR unlowered. Owner: #680.
- #680 multi-output re-consumption: record `push`, `wait`, and dependent
  consumer operations in `OutputPublicationPlan` when they precede the final
  output-store insertion anchor. Relocate the recorded sequence after the
  created compute in source order. This preserves the #666 no-deadlock
  invariant without rewrite-time analysis. Owner: #680 when refreshed onto
  merged #704.
- #680 tests and utilities: apply current Black and clang-format output and
  replace malformed near-split-marker comments that warn during MLIR tests.
  Owner: #680.
- #687 planning sources: apply current clang-format output. Owner: #687 when
  refreshed onto merged #704.
- #680 with the accumulation stack: retain DFB block subviews, inter-tile
  broadcasts, output ordering, and immutable planning. Owner: composition
  between #680 and the accumulation stack.
- #784 with #778 and #775: run PipeNet guard and schedule verification after
  synchronization insertion while logical DFB identities remain distinct, then
  coalesce acquires and finalize physical indices with `reuse-user-dfbs`.
  Owner: composition between independent stacks; no source PR change is
  required.
- #784 with #778: derive computed-address receiver backing descriptors from
  finalized `PhysicalDFBConfig` metadata or its six-field serialized form.
  Exclude separately allocated backing tensors from the static DFB budget while
  retaining their descriptors and compiler-assigned indices. Owner: composition
  between independent stacks; no source PR change is required.
- #784 with #778: use the internal logical-identity analysis header from its
  transform implementation directory and update verifier tests to require
  `dfb_id` before physical allocation. Owner: composition between independent
  stacks; no source PR change is required.
- #734 terminology: retain the multicast-to-scatter frontend and tests from
  the reviewed source change. Owner: #734.

## Demo Checklist

| Item | Implementation | Galaxy validation |
|---|---|---|
| Baseline K-sharded activation all-gather and N-sharded matmul | Pending restore | Pending |
| 1: N-sharded row-broadcast bias | Pending restore | Pending |
| 2: full-grid multi-node scheduling | Saved work pending replay | Pending |
| 3a: ReLU | Not started | Not started |
| 3b: GELU and SiLU | Not started | Not started |
| 4: chunked N output | Not started | Not started |
| 5: addcmul | Not started | Not started |
| 6: SwiGLU | Not started | Not started |
| 7: transpose selection | Not started | Not started |
| 8: fabric worker configuration | Not started | Not started |
| 9: FSDP weight gather | Not started | Not started |

## Validation

Current rebuild:

- Host build: passed through the current #782-#784 PipeNet stack.
- Pre-commit: all hooks passed through the current #782-#784 PipeNet stack.
- MLIR: `ninja -C build check-ttlang-mlir` passed on 2026-08-02; all 248 tests
  passed.
- Python-only integration tests: `test_compiler_options.py` and
  `test_kernel_runner.py` passed on 2026-08-02; all 61 tests passed.

Pending:

- Integrate the current grouped transport, fabric, selected PipeNet, and
  device-domain branches from `/home/bnorris/tt/PRs.md`.
- Restore the demo examples.
- Docker build and `check-ttlang-all`.
- Four-device example validation.
- Galaxy validation of each completed checklist item.
