# Demo Integration Branch

Target branch: `bnorris/integration-20260806`.

This branch combines unmerged development for validation and demos. It is not
a parent for source PRs.

## Integration Invariant

Rebuild from `origin/main` by merging every current leaf listed below. Do not
merge or cherry-pick intermediate PRs separately; each leaf already includes
its stack ancestry.

As of 2026-08-06, the current integration inputs are:

| Order | PR | Remote branch | Verified tip | Includes |
|---:|---:|---|---|---|
| 1 | #651 | `origin/bnorris/accumulation-scope-redesign` | `4fdff9d8` | Accumulation and DFB stack |
| 2 | #795 | `origin/bnorris/pr734-operation-device-domain` | `2dc25d54` | Operation device domains; incorporated into #734 by replayed commits |
| 3 | #780 | `origin/bnorris/pipes-codegen-optimizations` | `489ccccd` | Grouped/stateful PipeNet transport optimization |
| 4 | #754 | `origin/bnorris/pipes-issue-628-code-size` | `caf4fed0` | Compact selected PipeNet lowering |
| 5 | #734 | `origin/bnorris/pipes-multidevice-integrated-poc` | `b388d180` | Multidevice/fabric PipeNet support and #795 operation device domains |
| 6 | branch | `origin/bnorris/tensor-backed-dfb` | `ae22d6fe` | Tensor-backed DFB storage and runtime configuration |
| 7 | #803 | `origin/bnorris/preserve-subtile-dimensions` | `3061dcf5` | Subtile dimensions and compute-target validation |

#780, #754, and #734 are sibling leaves above #700. #651 is the accumulation
and DFB aggregate. Tensor-backed DFB is an independent branch based on current
main. #803 is an independent branch targeting main. Merge all six source
branches independently. #795 is listed for traceability but is not a separate
leaf. Restacking removed #795 as a Git
ancestor of #734, but #734 includes the operation-device-domain changes through
merge commit `0268ec34` and the corresponding replayed commits.

Before every rebuild:

1. Read `/home/bnorris/tt/PRs.md` and identify all current leaves.
2. Query each leaf PR for its live head and base branch.
3. Fetch all leaf refs and record their exact SHAs in this file.
4. Create and push a dated backup of the current integration branch.
5. Create a clean branch from the latest `origin/main`.
6. Merge each leaf with history. Do not reconstruct stacks commit by commit.
7. Resolve only conflicts between leaf tips. Record every resolution below.
8. Build and run `check-ttlang-mlir` locally.
9. Build in Docker and run `check-ttlang-all` before any Galaxy validation.
10. Restore demo-only commits after the source integration passes.

Name active integrations `bnorris/integration-YYYYMMDD`. Before replacement,
push `backup/integration-before-<change>-YYYYMMDD` at the current validated
head. Do not encode one constituent PR in the integration branch name.

## Current Rebuild

Branch: `bnorris/integration-20260806`.

Base: `origin/main` at `441be2d0`.

Backups:

- `backup/integration-before-pr803-remote-20260806` at `1c9d8f5e`
  (pushed).
- `backup/integration-before-tip-refresh-20260806` at `ff020b9d` (pushed).
- `backup/integration-before-tensor-backed-dfb-20260806` at `f9321102`
  (pushed).
- `backup/demo-before-main-refresh-20260805` at `c113b180` (pushed).
- `backup/demo-rebuild-v5-before-main-refresh-20260803` at `5e4c8cd7`
  (pushed).
- `backup/demo-rebuild-v6-before-tip-refresh-20260803` at `09bf2fe3`
  (pushed).
- `backup/demo-rebuild-v6-pr780-validated-20260803` at `13111c35`
  (pushed).
- `backup/demo-rebuild-v7-pr754-validated-20260803` at `193c81f3`
  (pushed).
- `backup/demo-rebuild-v7-pr734-validated-20260803` at `f5a92356`
  (pushed).
- `backup/demo-rebuild-v7-check-all-validated-20260803` at `a34ca2aa`
  (pushed over HTTPS after the GitHub SSH connection timed out).

| Leaf | Integration status |
|---|---|
| #651 | Current tip `4fdff9d8` merged in checkpoint `23c13dc0` |
| #795 | Not merged separately; current tip `2dc25d54` is represented by the operation-domain changes integrated into #734 |
| #780 | Current tip `489ccccd` merged in checkpoint `a3deca3a` |
| #754 | Current tip `caf4fed0` merged in checkpoint `8849a973` |
| #734 | Current tip `b388d180` merged in checkpoint `41c464a2` |
| Tensor-backed DFB | Current tip `ae22d6fe` merged in checkpoint `050ddd4b` |
| #803 | Current tip `3061dcf5` merged in checkpoint `43fe371c` |
| Demo-only commits | Base checkpoint restored as `f95ff833`; row-broadcast checkpoint restored as `befca9df`; module-command documentation restored as `9cb84198`; operation-domain test adaptation restored as `db1b54f8`; full-grid work pending |
| Integration fixes | Cross-leaf contracts preserved in `f6c02b39`; verifier contracts preserved in `1c9d8f5e` |
| Tensor-backed row-page views | 1x32 tensor storage may be interpreted directly as 16x32 or 32x32 compute pages in `10e3c3aa` |

## Integration Resolutions

These are composition resolutions unless a source owner is stated explicitly.
Any source defect found during validation must be recorded here before changing
the integration branch.

| Leaves | Resolution | Source action |
|---|---|---|
| `main` + #651 | Run PipeNet guard and schedule verification while logical DFB identities are distinct, before DFB coalescing and physical index finalization. | None; composition preserves logical-identity verification and reuse-aware finalization. |
| `main` + #651 | Add `dfb_id` to handwritten verifier test IR now checked by `DFBLogicalIdentityAnalysis`. | None; production IR already contains this required identity. |
| #651 + #780 | Update #780 runner tests from the former `(shape, ...)` serialized DFB tuple to #651's finalized `(num_tiles, ...)` runtime tuple. | None; integration-only test adaptation to the finalized runtime ABI. |
| #651 + #780 | Add logical `dfb_id` attributes to #780's handwritten full-pipeline transport test. | None; the production frontend already emits logical DFB identities. |
| #780 + #754 | Extend `ttl.pipe_transfer.create` to accept static or selected pipe references while retaining block-span and destination-depth transport metadata. `expectedReceivers` remains an optional static-pipe assertion and is prohibited for runtime-selected records. | None; this is the combined contract of sibling leaves. |
| #780 + #754 | Move selected-pipe copy expansion into #780's shared immutable `PipeTransferExpansionPlan` and remove #754's duplicate expansion implementation from `ConvertTTLToTTKernel.cpp`. | None; this removes integration duplication without changing either independent source branch. |
| #780 + #754 | Leave modules containing high-level PipeNet foreach callbacks unchanged in `ttl-form-pipe-transports`; `convert-ttl-to-ttkernel` lowers record selection and its typed control metadata together. Static PipeNets retain grouped transport. | None; grouping selected records would require a persistent cross-pass representation of `PipeForeachLoweringInfo`. |
| #780 + #754 | Update transport DFB analysis to require one transfer node through #754's plural record-aware graph API. | None; the grouping pass handles static transfers, which have one node per protocol operation. |
| #780 + #754 | Share page-granular unicast write emission between static and selected NoC transport emitters. | None; this satisfies the common transport interface and avoids duplicated write-loop construction. |
| #780 + #754 | Represent one selected protocol operation as one transport stream per matching record. The singular stream accessor remains restricted to static protocol operations. | Required when either stack is rebased onto the other; #780's operation-to-stream map must preserve every #754 record stream. |
| #780 + #754 | Compact receiver-published address-table entries from both static and selected resources. Both mechanisms share the per-core address table and contribute to `ttl.pipe_sram_scratch_bytes`. | Required when either stack is rebased onto the other; #780 resource finalization must include #754 `selectedResources`. Omitting them reduced scratch from 32 bytes to zero and deadlocked selected forwarding. |
| #651 + #754 | Add logical DFB identities and finalization metadata to #754 handwritten guard-verifier tests. Elide explicit checks for the implicit `ttl.yield` terminator and preserve data-flow ordering in lowering checks. | None; these are test-IR adaptations to #651 invariants and the existing custom assembly format. |
| #780 + #734 | Represent fabric as an explicit synchronization protocol in the backend-independent transport plan. Fabric transfers retain computed receiver addresses and use the existing static PipeGraph resource allocation. | Required when #734 is rebased onto #780; the fabric emitter must consume #780's transport plan rather than bypass it. |
| #780 + #734 | Propagate `DeviceTransferAttr` through #780's shared immutable high-level transfer-expansion plan and verify it on `ttl.pipe_transfer.create`. | Required when #734 is rebased onto #780; device transfer metadata must remain explicit after high-level pipe copies are expanded. |
| #754 + #734 | Treat values of selected-pipe record types, including foreach block arguments, as local pipe origins during device-transfer provenance analysis. | Required when #734 is rebased onto #754; checking only defining `ttl.select_pipe_*` operations rejects valid foreach block arguments. |
| #754 + #734 | Preserve device-qualified execution locations and receiver DFB identities while assigning receiver address sequences for selected records. | Required when #734 is rebased onto #754; logical devices must not alias when records share node coordinates and DFB indices. |
| #780 + #754 + #734 | Use the record-aware PipeGraph protocol-operation API for fabric diagnostics, while requiring one graph transfer for static fabric operations. | Required when the three PipeNet leaves are combined; the singular #734 lookup is incompatible with #754's selected-record graph. |
| #780 + #734 | Update the private operation-dispatcher unit invocation for #780's L1 budget argument. | None; this is an integration-only test adaptation to the current private callback signature. |
| #780 + #734 | Update #734 FileCheck assertions for #780's hoisted route and completion-address setup, and update provenance diagnostics to describe the accepted selected-record origins. | None; these are test expectation updates for the combined lowering. |
| #780 + #754 | Finalize sender address storage across both static protocol resources and selected-record resource tables. A transfer may pair a selected send with a static receive post, and both representations must share the sender's address allocation. | Required when #780 and #754 are combined; #780's finalizer must visit `selectedResources` after #754 adds selected protocol operations. |
| #780 + #754 | Match the compact-lowering Python lit check to the selected-transfer loop's semaphore table instead of the first table lookup after receiver publication. | None; this updates a #754 code-generation expectation for #780's additional selected-resource tables. |
| #780 + #754 | Check packed selected-record tables before the generated transfer operations in the 992-edge foreach code-size test. | None; this updates #754's operation-order expectation for the combined lowering. |
| `main` + #780 + #754 | Retain current main's DFB allocation-footprint API in `ttl-form-pipe-transports`, and leave high-level PipeNet foreach modules unchanged until selected records are lowered with their record metadata. | None; this combines current DFB reuse with the established selected-record ownership boundary. |
| `main` + #780 + #754 + #734 | Preserve current main's producer-recurrence handling while combining grouped transports, selected-record graphs, and device-qualified fabric transfers in PipeGraph planning and lowering. | None; this is the combined contract of the current leaves. |
| #754 + #734 | Restore `ttl.pipe_transfer.create` verification for positive and exact static `expectedReceivers`, prohibit it on selected records, and verify static `deviceTransfer` consistency before the static-pipe early return. | Required when #754 and #734 are combined; the independent verifier sections otherwise omit each other's checks. |
| #780 + #734 | Verify hoisted route and completion-address materialization before device conditionals without requiring unrelated address setup to be adjacent to the device comparison. | None; this is a generated-order expectation for the combined lowering. |
| #780 + #754 | Verify selected sender semaphore tables before the generated NoC write. | None; this is a generated-order expectation for the combined lowering. |
| `main` + current PipeNet leaves | Preserve current main's `pipe-global-semaphores-only` contract through the shared PipeNet planning options and counter allocator. Capacity-counter allocation must continue under the same policy after completion and readiness allocation. | None; #765 is merged in main. The integration conflict resolution had retained its test but omitted the option and policy propagation. Corrected in `ff020b9d`. |
| Restacked #780 and #754 | Preserve the validated composition tree while applying the current restack cleanup: transfer-plan declaration order and removal of obsolete Python PipeNet helpers/tests. | None; source trees changed only by those five verified files. |
| #734 + tensor-backed DFB | Retain multidevice attribute verifiers and bindings, add `TensorBackingAttr` verification/bindings, use the existing TTL dialect module for both attribute families, and retain both kernel-runner test groups. | None; the source changes are additive and conflict only because both branches extend the same registration and test sections. |
| PipeNet leaves + #803 | Retain the selected-pipe and multidevice TTL attribute helpers while adding #803 compute-type helpers. | None; both sets of helpers are required by the combined source tree. |
| Tensor-backed DFB + #803 | Preserve tensor-backed storage segments while using #803 validated subtile metadata and descriptors. Exclude tensor-backed storage from the statically allocated dataflow-buffer budget. | None; tensor-backed descriptors reference existing L1 storage and must not reserve a duplicate allocation. |
| PipeNet leaves + #803 | Update PipeTransport allocation-size callers to #803's diagnostic API and report the exact allocation failure from the pass. | None; the source changes extend the same allocation utility contract. |
| #651 + #803 | Classify integration-only `ttl.tile_accumulate` as a floating-point elementwise-binary compute target. | None; it lowers to the floating-point destination-reuse LLK, which has no integer lowering. |
| Tensor-backed DFB + #803 | Allow validated 1x32 tensor storage to use 16x32 or 32x32 DFB compute pages. Preserve the tensor-owned address and replace only descriptor format metadata; validate the logical shard byte range before program construction. | Add the validated implementation to the tensor-backed DFB source branch after the RMSNorm comparison confirms the required workload contract. |

## Validation

Current 2026-08-06 rebuild:

- Host `ttlang-opt` build: passed at `43fe371c`.
- MLIR: `ninja -C build check-ttlang-mlir`; 322 passed at `10e3c3aa`.
- Docker build: passed in `bnorris-ird3-v1.1.7` at `43fe371c`.
- Docker subtile, tensor-backed DFB, physical DFB metadata, runtime-config,
  and kernel-runner tests: 433 passed in 142.07 seconds at `43fe371c`.
- Docker tensor-backed DFB, descriptor, physical-metadata, runtime-config, and
  kernel-runner regressions: 287 passed at `10e3c3aa`. This includes BF16 and
  FP32 1x32-to-16x32 zero-copy device correctness and byte-range rejection.
- Simulator tensor-backed DFB signature tests: 2 passed at `10e3c3aa`.
- Targeted tensor-backed DFB and PipeNet hardware pytest: 189 passed and one
  expected failure in `bnorris-ird2-v1.1.7`.
- Docker `check-ttlang-all`: not yet rerun at `10e3c3aa`. The previous attempt's
  MLIR, Python bindings, and packaging passed. Its
  hardware pytest target encountered a sysmem mapping conflict after a separate
  fabric test acquired the device. Rerun after that test releases the device.
- Four-device and Galaxy demo validation: pending until `check-ttlang-all`
  passes.

Previous validated 2026-08-05 rebuild:

- Host build: passed.
- MLIR: `ninja -C build check-ttlang-mlir`; 304 passed.
- Docker build: passed in `bnorris-ird2-v1.1.7`.
- Docker `check-ttlang-all`: passed at `f9321102`.
  - MLIR: 304 passed.
  - Python bindings: 3 passed.
  - Packaging: 162 passed on both invocations.
  - Hardware pytest: 2,190 passed, 23 skipped, and 8 expected failures.
  - ME2E: 868 passed and 35 expected failures.
  - Python lit: 91 passed and one unsupported.
- Four-device and Galaxy demo validation: pending on this rebuild.

Prior validated integration (#651 + #780):

- Host build: passed.
- Docker Python unit subset: 77 passed.
- Docker PipeNet device subset: 12 passed.
- MLIR: `ninja -C build check-ttlang-mlir`; 273 passed.
- Changed Python files: Black 25.11 passed sequentially.
- Other all-file pre-commit hooks: passed.

#754 composition validation:

- Host and Docker builds: passed.
- Docker collective and forward-chain tests: 8 passed.
- Docker liveness-based resource-allocation program: passed with result
  verification.
- Compile-only 4x8 all-to-all: passed and reported exactly 992 edges.
- MLIR: `ninja -C build check-ttlang-mlir`; 282 passed.
- All-file pre-commit hooks: passed.

Current #734 composition validation:

- Host build: passed.
- Docker operation-domain, kernel-runner, and PipeNet validation tests: 109
  passed.
- Docker unicast capacity protocol tests: 8 passed, including the selected
  foreach regression that initially exposed selected-record provenance.
- MLIR: `ninja -C build check-ttlang-mlir`; 289 passed.
- Pre-commit hooks on every integration-resolution file: passed.
- Four-device `test_ccl.py::test_point_to_point[bf16]` did not reach TT-Lang
  compilation. Fabric initialization first reported a router handshake timeout;
  after a PCI reset, device enumeration failed in `ttnn.get_num_devices()` with
  `unordered_map::at`. The host reported `ETH_LIVE_STATUS=0x0`.

Final integration validation after the #734 merge and demo restoration:

- Docker `check-ttlang-all`: passed at `a34ca2aa`.
- MLIR: 289 passed.
- Python bindings: 3 passed.
- Packaging: 162 passed on both invocations.
- Hardware pytest: 2,139 passed, 23 skipped, 8 expected failures, and one
  unexpected pass.
- ME2E: 868 passed and 35 expected failures.
- Python lit: 89 passed and one unsupported.

Four-device and Galaxy validation remain pending for each completed demo
checklist item on the rebuilt integration.

The current four-device base-direct attempt at `c113b180` did not reach
TT-Lang compilation. Fabric initialization timed out on device 1 with four
router cores at `STARTED`. No process held any device before the run. A
subsequent `tt-smi -r all` completed, but `tt-smi -s` still reported
`ETH_LIVE_STATUS=0x0`; local fabric validation is blocked by hardware health.

Galaxy validation is blocked before allocation because `ssh exabox` times out
during the banner exchange to `127.0.0.1:22202`. Restore the laptop-managed
reverse tunnel documented in `/home/bnorris/tt/docs/Exabox.md`. The complete
candidate is pushed as `integration/demo-rebuild-20260803-v7` at `c113b180`;
`/tmp/tt-lang4-demo-c113b180.bundle` is also ready for direct transfer.

The first Docker `check-ttlang-all` attempt passed MLIR, bindings, packaging,
and the initial PipeNet hardware tests, then aborted in
`test_same_source_pipes_use_global_ready_counters` because the address finalizer
did not find the selected sender for a static loopback post. After generalizing
resource finalization, the exact hardware test passed, the complete
`test_pipenet_sync.py` file passed with 22 passes and one expected failure, and
the host MLIR suite passed 289 tests. The aggregate rerun passed 2,139 hardware
pytests with 23 skips, 8 expected failures, and one unexpected pass, then passed
868 me2e tests with 35 expected failures. Python lit then found an
over-constrained code-generation check in `pipenet_resource_allocation.py`;
the corrected configured test passes individually. The complete target then
passed 88 tests before finding a second operation-order expectation in
`pipenet_foreach_iteration.py`; that corrected configured test also passes
individually. The complete Python lit target now passes 89 tests with one
unsupported test. The final aggregate rerun passed all targets.

## Demo Checklist

| Item | Implementation | Four-device validation | Galaxy validation |
|---|---|---|---|
| Base direct and context all-gather minimal matmul | Restored in `f95ff833` | Blocked before compilation: `ETH_LIVE_STATUS=0x0` | Pending current tip |
| N-sharded row-broadcast bias | Restored in `befca9df` | Pending current tip | Pending current tip |
| Full-grid multi-node scheduling | Saved example-only work identified; compiler workaround excluded | Pending | Pending |
| ReLU | Not started | Pending | Pending |
| GELU and SiLU | Not started | Pending | Pending |
| Chunked N output | Not started | Pending | Pending |
| Addcmul | Not started | Pending | Pending |
| SwiGLU | Not started | Pending | Pending |
| Transpose selection | Not started | Pending | Pending |
| Fabric worker configuration | Not started | Pending | Pending |
| FSDP weight gather | Not started | Pending | Pending |
