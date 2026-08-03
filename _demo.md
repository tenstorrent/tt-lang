# Demo Integration Branch

Branch: `bnorris/demo`.

Base: `origin/main` at `90d5f0a8`.

Purpose: aggregate current unmerged compiler work for integration testing and
the all-gather minimal matmul examples. This branch is not a source PR parent.

## Update Policy

- Maintain source PR branches independently using
  `/home/bnorris/tt/doc/StackedPRs.md`.
- Fetch source refs and create a dated, pushed backup before rebuilding.
- Rebuild from `origin/main` when constituent branches have been rewritten.
- Restore demo-only commits only after all source branches are integrated.
- Record every integration repair below, including its source owner and the
  required source-branch action.
- Run Docker `check-ttlang-all` after a major rebuild and before Galaxy tests.
- Validate each demo checklist item on Galaxy before starting the next item.

## Current Rebuild

Started: 2026-08-02.

Backup: `backup/demo-before-rebuild-20260802-192524` at `db9d2dfb` (pushed).

| Order | PR or branch | Source tip | Status |
|---:|---|---|---|
| 1 | `origin/main` | `90d5f0a8` | Applied |
| 2 | accumulation aggregate (#704, #733, #778, #775) | `fba66fa8` | Applied |
| 3 | #673 indexed PipeNet receivers | `713892fa` | Applied |
| 4 | #780 grouped PipeTransport | `b544c620` | Applied |
| 5 | #687 control-flow stores | `a4ca5487` | Applied |
| 6 | #680 DFB subviews | `4588cf72` | Applied |
| 7 | #734 multidevice fabric | `9acae9c1` local reviewed tip | Applied |
| 8 | #754 compact selected PipeNets | `5cde0870` | Applied |
| 9 | #795 operation device domains | `72a343c3` | Applied |
| 10 | demo examples | `a3021dca`, `37c570f5` | Applied |

## Integration Repairs And Source Follow-Ups

Each item must either be applied to the named source PR or remain identified as
an aggregate-only composition change between independent PRs.

- #780 with the accumulation stack: preserve accumulation-scope lowering before
  loop-state lowering; run PipeNet synchronization before DFB accumulation;
  finalize physical DFB allocation only after logical DFB identity analysis.
  Owner: aggregate ordering between #704/#733/#778/#775 and #780.
- #687 with #704: port multi-block store replacement to immutable
  `IntermediateDFBPlanning`; do not restore mutation during analysis. Owner:
  aggregate composition between #687 and #704.
- #680 with the accumulation stack: retain DFB block subviews, inter-tile
  broadcasts, output ordering, and the current immutable planner. Owner:
  aggregate composition between #680 and the accumulation stack.
- #734 with #780: implement fabric as a `PipeSynchronizationProtocol` selected
  by the current transport plan; use grouped packetization payload sizes and
  transport-owned storage. Owner: #734 must be restacked after #780 is final.
- #734 receiver identity: qualify physical receiver DFB keys and schedule
  execution points by `DeviceTransferAttr`; retain immutable
  `PipeTransferIndex` and DFB release analysis. Owner: #734.
- #734 provenance: resolve device transfers through private helper call sites
  and structured control flow in `TransferProvenance`; keep high-level copy
  expansion in `PipeTransferExpansion.cpp`. Owner: #734.
- #734 obsolete commits: skip the old receiver-stream index and formatting-only
  planner commits because #780 supplies `PipeTransferIndex`, immutable DFB
  analysis, and newer planner APIs. Required semantics were ported explicitly.
- #734 terminology: retain the multicast-to-scatter frontend and tests from
  `5ac0c3d7`. Owner: #734.
- #754 with #780/#734: represent protocol operands as static-or-selected pipe
  references while preserving grouped block span, destination group depth,
  fabric synchronization, and device-qualified receiver DFB identity. Owner:
  aggregate composition until #754 is restacked over #780 and #734.
- #754 conversion ownership: keep record-loop selection in
  `ConvertTTLToTTKernel.cpp`; keep high-level pipe-copy expansion in
  `PipeTransferExpansion.cpp`. Do not retain duplicate expansion code. Owner:
  #754 when restacked over #780.
- #754 replay completeness: restore the four-record direct-lowering threshold,
  temporary selected-op legality, and dead selected-op cleanup from the source
  branch. These were omitted while resolving the conversion ownership
  conflict. Owner: demo replay only; no source PR change is required.
- #754 with #780 transfer schema: `expectedReceivers` is unused by transport
  planning and cannot describe records whose receiver counts are selected at
  runtime. Remove the redundant attribute and derive receiver counts from the
  static or selected pipe reference. Owner: aggregate composition; #754 should
  remove the attribute when restacked over #780.
- #754 selected transfer expansion: preserve selected pipe references through
  `PipeTransferExpansion.cpp`, derive the PipeNet id from `PipeReference`, and
  treat selected records as local transfers unless their IR gains explicit
  device-transfer metadata. Owner: aggregate composition between #754 and
  #734.
- #754 with #780 implementation: commit `8e96ef7b` removes the redundant
  receiver-count attribute, restores selected-pipe expansion, retains grouped
  transfer attributes, and adds selected device-transfer validation. Owner:
  aggregate composition; apply the equivalent change when #754 is restacked.
- #795 with the unified-operation updates: preserve both `device_domain` and
  the newer `l1_budget_override` argument in `python/ttl/atom.py`. Owner:
  aggregate composition between #795 and the current operation dispatcher.

## Demo Checklist

| Item | Implementation | Galaxy validation |
|---|---|---|
| Baseline K-sharded activation all-gather and N-sharded matmul | Restored in `bc438ec1` | Pending |
| 1: N-sharded row-broadcast bias | Restored in `39a0b1f6` | Pending |
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

Pending for this rebuild:

- Host build and MLIR tests.
- Docker build and `check-ttlang-all`.
- Four-device example validation.
- Galaxy validation of each completed checklist item.
