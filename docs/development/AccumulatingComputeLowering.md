# Accumulating Compute Lowering

This document describes how the tt-lang compiler lowers operations that
accumulate results across multiple invocations - reductions, matmul
K-accumulation, and user-written `+=` loops - onto the Tenstorrent
compute engines.

## Overview

An accumulation in tt-lang can be compiled in three ways with the same
program semantics and different thread-local data movement:

1. Keep the partial value in the destination register file (DST). This
   avoids intermediate L1 traffic while the computation fits in one acquired
   DST lifetime, and matches the preferred form in handwritten kernels when
   live DST capacity is sufficient.
2. Add packed output tiles into L1 through the packer. This uses hardware L1
   accumulation for additive recurrences that must persist across separate DST
   lifetimes.
3. Carry the partial value through an explicit compiler-managed dataflow
   buffer (DFB). This is the general lowering, but additive recurrences pay an
   L1 wait/store cost on each iteration. Handwritten kernels use scratch DFBs
   for intermediates that must be reused by later compute stages or cannot
   remain live in one DST lifetime.

DST residency and L1 packer accumulation use hardware accumulation
mechanisms. Explicit DFB state is the general lowering for recurrences that
cannot use either optimized form, such as values reused by later compute
stages or, in future conditional accumulation support, updates that require
explicit `has_value` state across dynamic control flow.

The compiler recognizes these accumulation forms and initial-state cases:

- `reduce_tile` and `matmul_tiles` accumulate per-tile over a reduction
  dim. Additive reductions can use DST or L1 packer accumulation.
  `reduce_max` remains DST-resident because L1 packer accumulation is
  additive.

- User-written `out_blk += ...` loops lower to L1 packer accumulation.
  The lowering records whether iteration 0 overwrites or accumulates onto
  a prior output value. The store-then-accumulate pattern
  (`out_blk.store(v); for K-1: out_blk += ...`) is represented as
  accumulation onto an existing value.

- Loop-carried additive recurrences inside an `scf.for` (`acc = acc +
  x` or `acc = x + acc`, plain tensor target) use accumulation-scope IR
  before general tensor state materialization. Strategy lowering compares
  legal DST and L1 packer candidates with the accumulation cost model.

- General tensor recurrences that are not recognized as additive
  accumulation use explicit compiler-managed DFB state.

The accumulation-scope IR declares which destination tensor views participate
in an accumulation region, plus the initial-state policy for each output. Later
lowering can select DST, L1 packer accumulation, or explicit DFB state without
reconstructing that policy from neighboring stores or DFB operations.

The rest of this document details each piece: accumulation scopes,
loop-carried tensor state elimination (`ttl-materialize-loop-state`),
`DstSectionOp` as the IR primitive that keeps DST live, the choice between
DST, L1, and explicit DFB state materialization, the emitted loop structure,
per-op init insertion, and L1 accumulation reconfiguration placement.

## Implemented Semantics and Deferred Features

The current implementation supports additive accumulation with two optimized
storage strategies, while general tensor state still uses explicit DFB
materialization:

- `auto`, `dst`, and `l1-pack` strategy selection for tensor additive
  recurrences;
- L1 packer metadata lowering for explicit DFB `+=`;
- reduction-capable `ttl.compute` lowering through the same accumulation
  metadata contract;
- metadata-driven TTKernel L1 reconfiguration insertion using
  `ttl.l1_acc_initial` and `ttl.l1_acc_scope_id`.

Deferred features are tracked separately:

- #640: optimize additive tensor recurrences with post-loop pure users;
- #645: add source-level accumulation strategy hints;
- #646: define non-additive accumulation update contracts;
- #648: define nested and conditional accumulation scope semantics;
- #649: replace `maximize-dst` with granular compiler options;
- #650: synthesize explicit DFB state fallback for unsupported scopes.
- #652: add explicit matmul K-accumulation strategy selection.

The current design preserves these invariants:

- `ttl.accumulation_scope` declares accumulation outputs and policies. It
  does not encode DST, L1 packer, or explicit DFB state.
- Conditional rejection belongs in `ttl-insert-accumulation-scopes{kind=dfb}`,
  not in
  the `ttl.accumulation_scope` verifier. The verifier remains structural.
- `ttl.l1_acc_loop` plus `ttl.l1_acc_initial` is static first-update
  lowering metadata, not the full accumulation model.
- TTKernel passes must not infer accumulation semantics from neighboring DFB
  or pack operations.
- Future conditional support should add update-site metadata or explicit
  lowered state transitions before TTKernel L1 insertion.

## Accumulation Scope IR

`ttl.accumulation_scope` declares the accumulation contract for one or more
destination tensor views. It records which outputs share a region, how each
output is initialized, and which value returned by the region updates each
output. The op does not select the storage mechanism used for partial values.
It has:

- `outputs`: destination tensor views governed by the accumulation policy;
- `inits`: init operands for outputs whose initial mode is `init`;
- `initial_modes`: one accumulation initial-mode per output (`overwrite`,
  `accumulate_existing`, or `init`);
- `body`: a single-block region with one block argument and one yielded value
  per output.

The op has `RecursiveMemoryEffects`; its effects are the effects of the
body. It produces no tensor results. Tensor recurrence scopes are consumed
before general loop-state materialization, so value-style accumulation is a
deferred feature rather than part of the current op contract.

The verifier is structural:

- initial-mode count equals output count;
- init modes have matching init operands;
- init operand types match their corresponding outputs;
- the body has one block argument and one yielded value per output;
- body arguments and yielded values match their output types;
- nested `ttl.accumulation_scope` is rejected until nested accumulation
  semantics are defined.

The verifier does not prove that stores target the declared outputs or that
control flow reaches an update. Those are nonlocal insertion and strategy
lowering responsibilities.

Initial modes have these meanings:

- `overwrite`: the first executed contribution defines the accumulator
  value. Current L1 lowering supports only the static case where that first
  contribution is unconditionally ordered.
- `accumulate_existing`: an existing value in the output location
  participates in the result. For L1 packer accumulation, iteration 0 must
  pack with L1 accumulation enabled.
- `init`: an init operand seeds the accumulator, independent of the final
  output location.

Tensor recurrence scope form:

```mlir
ttl.accumulation_scope
    outs(%out_view : tensor<...>)
    inits(%init : tensor<...>)
{
^bb0(%state: tensor<...>):
  %result = scf.for ... iter_args(%acc = %state) -> tensor<...> {
    %next = ttl.add %acc, %contribution : tensor<...>
    scf.yield %next : tensor<...>
  }
  ttl.store %result, %out_view : tensor<...>, tensor<...>
  ttl.yield %result : tensor<...>
} initial_modes([init])
```

Accumulation scopes expose accumulator state as block arguments and return the
updated state through `ttl.yield`. Cross-output dependence is represented by
ordinary SSA use-def edges between yielded values.

Multi-output stateful accumulation scope form:

```mlir
ttl.accumulation_scope
    outs(%out0, %out1 : tensor<...>, tensor<...>)
    inits(%init0, %init1 : tensor<...>, tensor<...>)
{
^bb0(%acc0: tensor<...>, %acc1: tensor<...>):
  %next0 = ttl.add %acc0, %acc1 : tensor<...>, tensor<...> -> tensor<...>
  %next1 = ttl.add %acc1, %next0 : tensor<...>, tensor<...> -> tensor<...>
  ttl.yield %next0, %next1 : tensor<...>, tensor<...>
} initial_modes([init, init])
```

This form exposes accumulator state as block arguments and returns the updated
state through `ttl.yield`. Cross-output dependence is represented by ordinary
SSA use-def edges between yielded values. This covers state updates such as
max-plus-rescale recurrences that are not additive recurrences.

Explicit DFB accumulation scope form:

```mlir
ttl.accumulation_scope
    outs(%out_view : tensor<...>)
{
^bb0(%state: tensor<...>):
  scf.for ... {
    ttl.store %value, %out_view {accumulate} : ...
  }
  ttl.yield %state : tensor<...>
} initial_modes([overwrite])
```

`AccumulationScopeOpInterface` is implemented by `ttl.accumulation_scope`
and reduction-capable `ttl.compute`. Consumers call `isAccumulation()` before
reading outputs, inits, initial modes, and the accumulation body. This
keeps reduction L1 metadata and `ttl.accumulation_scope` L1 metadata on one
contract without forcing reductions to be wrapped in `ttl.accumulation_scope`.

## Pipeline Ownership

The TTL-to-TTKernel pipeline handles accumulation in this order:

1. `ttl-form-accumulation-scopes{strategy=<accumulation-strategy>}` runs before
   `ttl-materialize-loop-state`. It forms semantic scopes around recognized
   single-output additive tensor recurrences when at least one legal lowering
   exists for `accumulation-strategy`, and records `init` initial mode.

2. `ttl-lower-accumulation-scopes{strategy=<accumulation-strategy>}` consumes
   those scopes. It selects DST or L1 packer accumulation according to
   `accumulation-strategy`. Stateful scopes with yielded state lower in `auto`
   mode by emitting one final `ttl.store` per yielded value and leaving tensor
   loop-carried state for explicit DFB materialization. Required `dst` or
   `l1-pack` strategy reports an error for stateful scopes until stateful DST
   or L1 packer lowering is implemented. The pass removes the semantic wrapper
   before general loop-state materialization.

3. `ttl-materialize-loop-state` handles remaining tensor `scf.for`
   iter_args through compiler-allocated DFB state. Additive recurrences
   recognized by scope insertion do not reach this pass.

4. `ttl-insert-copy-wait` and `ttl-auto-sync` run before DFB `+=`
   detection, so scope insertion sees canonical DFB acquire/release
   structure and required copy waits.

5. `ttl-insert-accumulation-scopes{kind=dfb}` inserts semantic scopes around
   user-written accumulating stores. It computes `overwrite` versus
   `accumulate_existing` before any TTKernel conversion.

6. `ttl-lower-accumulation-scopes{kind=dfb}` consumes DFB scopes and emits
   L1 packer metadata. DFB scopes always lower to L1 in the current
   implementation because the source construct updates an output block, not a
   loop-carried tensor value.

7. `ttl-create-producer-compute`, `ttl-insert-intermediate-dfbs`, compute
   conversion, DST assignment, optional subblocking, and `ttl-lower-to-loops`
   run. Reduction-capable `ttl.compute` uses
   `AccumulationScopeOpInterface` to emit L1 metadata when it lowers to the
   L1 reduction form.

8. After TTL-to-TTKernel conversion, `ttkernel-insert-l1-accumulation`
   consumes L1 metadata and inserts `pack_reconfig_l1_acc` operations.

Insertion and lowering are adjacent for each scope kind. Do not insert
canonicalization or CSE between them; the scope body is a temporary semantic
region, not a long-lived optimization boundary.

## Strategy Selection and Legality

The public `accumulation-strategy` option currently accepts `auto`, `dst`,
and `l1-pack`. It controls tensor recurrence scopes only. Explicit DFB
accumulation scopes always lower to L1 packer metadata; `dst` is ignored for
DFB scopes because DFB state cannot reside in DST.

| Source form | `auto` | `dst` | `l1-pack` |
| --- | --- | --- | --- |
| Tensor additive recurrence | DST if legal, otherwise L1 packer accumulation | require DST; diagnose if illegal | use L1 packer accumulation |
| Explicit DFB `+=` | L1 packer accumulation | L1 packer accumulation | L1 packer accumulation |
| Additive reductions in `ttl.compute` | Existing `maximize-dst` / `dst-accumulation` policy | Existing `maximize-dst` / `dst-accumulation` policy | Existing `maximize-dst` / `dst-accumulation` policy |

For `accumulation-strategy=auto`, selection is per
`ttl.accumulation_scope`. A scope with independent output groups may be
planned per `AccumulationGroup` once grouped lowering supports partial scope
rewrites. A dependent group is selected as one unit because lowering one
accumulator independently can break SSA dependences between accumulator
updates.

`AccumulationGroupAnalysis` records output slots, state values, and
cross-output dependences. `planTensorAccumulationStrategy` enumerates legal
strategy candidates for each group and scores them with
`AccumulationCostModel`. The cost model selects architecture-specific weights
from `ttl.target_arch` when present. It estimates:

- one-time and per-iteration DFB hops;
- one-time and per-iteration pack/unpack tile traffic;
- live DST tiles required by the selected strategy;
- packer reconfiguration count.

When the loop trip count is statically known, the score is:

```text
total_dfb_hops = one_time_dfb_hops +
                 iterations * per_iteration_dfb_hops
total_pack_unpack_tiles = one_time_pack_unpack_tiles +
                          iterations * per_iteration_pack_unpack_tiles
estimated_cost = total_dfb_hops * dfb_hop_fixed_cost +
                 total_pack_unpack_tiles * dfb_hop_per_tile_cost
```

If the trip count is not known, the model still records the feature counts but
does not compute `estimated_cost`. Candidate comparison then uses the feature
counts directly in this order: per-iteration DFB hops, one-time DFB hops,
per-iteration pack/unpack tile traffic, one-time pack/unpack tile traffic,
packer reconfiguration count, then live DST tiles.

If the target architecture is absent or not calibrated, `estimated_cost` is
also reported as `unknown`, and the same feature-count ordering is used.

Cost model decisions are printed by the `ttl-lower-accumulation-scopes` debug
stream:

```bash
ttlang-opt input.mlir \
  --pass-pipeline='builtin.module(func.func(ttl-lower-accumulation-scopes{kind=tensor strategy=auto}))' \
  -debug-only=ttl-lower-accumulation-scopes 2>&1
```

The debug output lists the target architecture, every legal or rejected
candidate, the estimated score when available, each feature count, and the
selected strategy:

```text
accumulation cost model target_arch=wormhole_b0
  candidate strategy=dst legal=true estimated_cost=2546 ...
  candidate strategy=l1-pack legal=true estimated_cost=2954 ...
  selected strategy=dst
```

The current tensor strategy candidates populate the features as follows:

| Candidate | Feature model |
| --- | --- |
| DST-resident tensor accumulation | One initial accumulator materialization, streamed per-iteration contribution traffic or one resident contribution materialization, and live DST tiles equal to one accumulator tensor. |
| L1 packer tensor accumulation | One initial output materialization, two DFB handoffs per iteration for accumulator read/update through L1, per-iteration pack/unpack traffic for accumulator update, and two packer reconfigurations. |

The cost values are calibration inputs, not semantic requirements. They affect
profitability only; legality is checked before a candidate is scored.

Blackhole fixed and per-tile values come from direct flash MLA handoff
ablation. Wormhole values are relative scores derived from the matched
Wormhole and Blackhole LLK perf artifacts in the `tenstorrent/tt-metal`
`LLK perf` scheduled workflow run `27594326478` (run number 63), created on
June 16, 2026 UTC from `main` commit
`393e4c9909abd8c589bb269e6a93571151bcf1c7`. The run's
`perf-data-wormhole-*` and `perf-data-blackhole-*` artifacts contain
postprocessed CSVs for the same LLK suites. Their `TILE_LOOP` medians give
about `1.361x` for `L1_TO_L1` and `2.036x` for `UNPACK_ISOLATE`; applying
those ratios to the Blackhole handoff fit gives the rounded Wormhole scores
below.

| Cost input | Current use | Source |
| --- | --- | --- |
| Blackhole DFB fixed handoff cost | `dfbHopFixedCost = 210` | Track A flash MLA shard ablation from Zoe Carver's June 2026 Slack benchmark notes in `#tt-lang`; measured about 0.21 us of fixed cost per DFB handoff. |
| Blackhole DFB per-tile traffic cost | `dfbHopPerTileCost = 67` | Track A flash MLA shard ablation from Zoe Carver's June 2026 Slack benchmark notes in `#tt-lang`; fit DFB handoff time as about `0.21 us + 0.067 us * tile_count` for 32x32 tiles. |
| Wormhole DFB fixed handoff score | `dfbHopFixedCost = 286` | `tenstorrent/tt-metal` `LLK perf` scheduled workflow run `27594326478` from June 16, 2026 UTC, on `main` at commit `393e4c9909abd8c589bb269e6a93571151bcf1c7`; the `TILE_LOOP` `L1_TO_L1` median ratio between `perf-data-wormhole-*` and `perf-data-blackhole-*` artifacts is about `1.361x`, so `210 * 1.361` rounds to `286`. |
| Wormhole DFB per-tile traffic score | `dfbHopPerTileCost = 136` | Same tt-metal LLK perf run; the `TILE_LOOP` `UNPACK_ISOLATE` median ratio is about `2.036x`, so `67 * 2.036` rounds to `136`. |
| DST-resident accumulation preference | Candidate scoring favors removing per-iteration DFB handoffs and pack/unpack traffic when DST legality holds. | Track C flashloop results from Zoe Carver's June 2026 Slack benchmark notes in `#tt-lang`; they show large wins from keeping dependent state in DST and removing repeated dataflow-buffer synchronization. |

Tensor DST lowering requires the normalized additive recurrence form:

- exactly one output and one init operand;
- `init` initial mode;
- output from `ttl.cb_reserve`;
- a top-level `scf.for` followed by the final non-accumulating `ttl.store`;
- recurrence dataflow matching `acc = acc + contribution` or
  `acc = contribution + acc`;
- no post-loop pure users between the loop and final store, except removable
  reserve/attach operations. #640 tracks fused epilogue-style support.

Tensor L1 packer lowering creates a non-accumulating store of the init
operand before the loop and per-iteration accumulating stores into the same
output reservation. The generated loop carries
`ttl.l1_acc_initial = accumulate_existing` because the pre-loop store has
already materialized the accumulator baseline in L1.

Reduction L1 lowering is additive only. `reduce_max` and
`ttl.tile_accumulate ... max` have no L1 packer representation in this
lowering, so they remain DST-resident. #646 tracks non-additive update
contracts and the shared legality table for recurrence classes and strategy
pairs.

Unsupported L1 output formats are diagnosed in
`TTKernelInsertL1Accumulation`, where the final pack output type is visible.
The supported formats are Float32, Float16, BFloat16, Int32, and UInt8,
matching the current TTKernel lowering allowlist.

### Matmul K-Accumulation (planned, #652)

Matmul K-accumulation (`C[M,N] += A[M,K] @ B[K,N]` over `kt` contraction
tiles) is the same DST-versus-L1 choice as a tensor additive recurrence; #652
will select it with `matmul-k-accumulation=auto|dst|l1-pack`. The two candidates
map to existing lowerings:

- DST K-accumulation keeps the `M*N` output tiles resident in one
  `ttl.dst_section` across all `kt` steps and packs once (the accumulating
  `DstSectionOp` placement with `matmul_block`, `kt_dim > 1`).
- L1 packer K-accumulation packs the `M*N` partial after each K step with
  packer L1 accumulation enabled.

At equal accumulation precision this choice is degenerate the same way the
additive recurrence is. A single matmul instruction writes its `m_sb x n_sb`
output partial into DST in both candidates (the matmul reads operands from the
separate srcA/srcB banks, not DST), so both tile the output into the same
`<= getDstCapacity` subblocks and re-read the `A` and `B` operands the same
number of times. DST accumulation then packs each output tile once; L1 packer
accumulation re-packs it every K step. So at equal precision DST has the same
operand traffic and fewer packs and always wins, matching the handwritten-kernel
result that one `matmul_block` accumulating in DST and packing once leaves the
packer idle while a per-step L1 pack saturates it.

The non-degenerate case is fp32 accumulation. A resident fp32 accumulator halves
DST capacity (`getDstCapacity`: 16 bf16 tiles in full-sync, 8 by default; 8 / 4
for fp32 -
[matrix engine report](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/tech_reports/matrix_engine/matrix_engine.md)),
so DST K-accumulation in fp32 must use a smaller output subblock, which raises
the operand re-read factor (`A` read `N / n_sb` times, `B` read `M / m_sb`
times) and the subblock count. L1 packer accumulation avoids this: it keeps DST
in bf16 - the larger subblock - and lets the packer add each bf16 partial into
an fp32 L1 accumulator (same report: "accumulate in fp32, then final output as
float16_b"). So the fp32 choice trades DST's extra operand re-reads against L1's
per-K-step fp32 packs, at a precision between full fp32-DST and bf16-DST
accumulation. This is the crossover the cost model must score; bf16 accumulation
does not reach it.

Candidate feature model in the fp32 crossover (output `P = M*N`, depth `kt`,
operands `A = M*kt`, `B = kt*N`):

| Candidate | Feature model |
| --- | --- |
| DST K-accumulation | Output packed once; operands read once times a re-read factor that grows as the fp32 subblock shrinks; live DST tiles equal one fp32 output subblock. |
| L1 packer K-accumulation | DST stays bf16 (larger subblock, fewer operand re-reads); output packed every K step into an fp32 L1 accumulator; two packer reconfigurations. |

The operand re-read is an explicit per-candidate feature, not a constant: each
candidate carries `M*N*kt * (1/m_sb + 1/n_sb)` operand-unpack tiles for its own
subblock `(m_sb, n_sb)`. It cancels when both candidates accumulate at the same
precision (identical subblock) and differs only when fp32 halves the DST
subblock, so it is exactly the term that decides the fp32 crossover. The score
must take `(m_sb, n_sb)` from the subblocker rather than assume it.

Scoring this requires splitting `dfbHopPerTileCost` into separate unpack, pack,
and L1-accumulation-pack components, because DST's extra cost here is operand
unpack and L1's is pack, and their Wormhole/Blackhole ratios differ. In the June
16, 2026 tt-metal LLK run `27594326478`, matmul operand unpack is about equal on
both parts (`UNPACK_ISOLATE` ~37.6 Blackhole, ~39.6 Wormhole, ~1.05x), a
destination pack is far costlier on Wormhole (`pack_dest_bank` `PACK_ISOLATE`
~36 versus ~86, ~2.40x), and an L1-accumulation pack adds ~30-35% over a plain
pack on Blackhole but is near-free on Wormhole (`pack_dest_bank`
`L1Accumulation` Yes versus No). A single per-tile cost cannot express a
crossover that pits an unpack-dominated cost against a pack-dominated one with
different per-engine ratios. Matmul MATH stays out: it is `kt * P` tile
operations in both candidates and the matrix engine is identical across Wormhole
and Blackhole (matrix engine report, above), so it cancels.

Effects this data-movement model does not capture, each a reason to prefer L1
that a per-tile-traffic score cannot see:

- DST accumulation holds the output subblock in DST across the entire K loop. It
  forces `dst_full_sync_en` for large subblocks (losing kernel-wide
  packer/compute overlap) and starves DST for fused neighbors - a matmul between
  softmax stages cannot keep its accumulator resident while the softmax needs
  DST. This occupancy cost can favor L1 even at bf16.
- Choosing `matmul_block` (one block instruction) over per-tile `matmul_tiles`
  for the same DST pack-once work is a per-call init effect, not pack/unpack
  traffic.
- The subblock re-read factor depends on the subblocker's tiling shape, which
  the score must read from the chosen subblocking rather than assume.

## Unsupported Structured Control Flow

Conditional DFB `+=` is rejected in `ttl-insert-accumulation-scopes`. The
current L1 metadata represents a static first-update policy: iteration 0
overwrites or accumulates according to `ttl.l1_acc_initial`. It cannot
represent "the first dynamically executed update overwrites, later executed
updates accumulate" when a pack may be skipped by an `scf.if`.

The same pass rejects a loop that contains both a DFB `+=` and a plain store.
TTL source-level lowering does not yet encode which in-loop stores should run
outside the accumulation lifecycle, so it cannot prove the required packer
state transitions before TTKernel conversion.

Future conditional support (#648) should add explicit update-site metadata
or lowered state transitions before TTKernel L1 insertion. The required
semantic state is:

```
if update_executes:
    if has_value:
        accumulate contribution
    else:
        overwrite with contribution
        has_value = true
```

Initial modes define the initial `has_value` state: true for `init` and
`accumulate_existing`, false for `overwrite`.

Nested `ttl.accumulation_scope` is rejected by the op verifier until #648
defines how nested scopes compose. Nested independent L1 packer scopes need
explicit state transitions on scope entry and exit; current TTKernel
metadata supports one active packer L1 accumulation lifecycle per lexical
loop nest.

## L1 Packer Metadata Contract

Strategy lowering and reduction lowering use these attributes after semantic
scope consumption:

- `ttl.l1_acc_loop`: marks a user or compiler-generated loop containing L1
  packer accumulation stores.
- `ttl.reduction_loop`: marks a loop generated from reduction-capable
  `ttl.compute` that lowered to L1 packer accumulation.
- `ttl.l1_acc_initial`: records `overwrite` or `accumulate_existing`.
- `ttl.l1_acc_scope_id`: groups loops that belong to one semantic
  accumulation lifecycle.

`TTKernelInsertL1Accumulation` validates that annotated loops have both
`ttl.l1_acc_initial` and `ttl.l1_acc_scope_id`. It groups adjacent loops by
scope id, validates accumulated-output pack formats, brackets in-loop packs to
non-scope DFBs, and inserts packer reconfiguration operations. It does not
infer initial mode from neighboring DFB reserve/push/store operations or from
TTKernel pack order.

Nested annotated loops with different scope ids are rejected until #648 adds
explicit packer state transitions.

## Loop-Carried Tensor State and Accumulation Scopes

A Python `for` loop that reassigns a tensor variable read on the next
iteration (`acc = acc + x`, `acc = relu(acc)`) compiles to an `scf.for`
with a ranked-tensor `iter_arg`. The accumulation pipeline first inserts
`ttl.accumulation_scope` around recognized additive recurrences. General
tensor recurrences that remain after strategy lowering are eliminated by
`ttl-materialize-loop-state` before compute lowering.

### Why Tensor `iter_args`, Not DFBs Directly

The frontend could emit DFB state directly from the AST and skip the
tensor `iter_arg` form. It does not, for the following reasons.

**Layering.** A rebound Python loop variable is a value carried to the
next iteration; a tensor `scf.for` iter_arg is its direct translation.
Emitting DFBs would force the AST walker to choose DFB identifiers, block
counts, and slot flow control, which are backend concerns.

**Strategy decided in MLIR.** Additive-vs-general classification depends
on use-def structure - the single add, its single use, the consumer
store, and the reserve feeding it - which
`matchAdditiveTensorAccumulation` matches reliably and the AST cannot.
The frontend stays a correctness-only component that identifies
loop-carried variables; MLIR scope insertion and strategy lowering
choose DST, L1 packer accumulation, or general DFB state.

**Shared state materialization.** Additive, elementwise, and tuple
recurrences are all tensor iter_args at the frontend. Additive
recurrences are consumed by accumulation strategy lowering, while
remaining tensor state uses the same reserve/store/wait/attach helper
code as `ttl-insert-intermediate-dfbs` through `DFBMaterialization`.

Tensor-level loops also remain subject to standard canonicalization, CSE,
and dead-code elimination, which do not apply to side-effecting DFB ops.

### Why Not One-Shot Bufferization

Upstream MLIR eliminates tensor `scf.for` iter_args with one-shot
bufferization: `scf::ForOp`'s `BufferizableOpInterface` implementation
threads each tensor iter_arg through the loop as a memref and drops the
tensor result. tt-lang does not bufferize tensors to memref. On-chip a
tensor value lives in the DST register file or in a dataflow buffer (DFB)
accessed through `cb_reserve`/`store`/`cb_wait`/`attach_cb`; neither is a
memref. This pass eliminates the tensor iter_arg by realizing the carried
state as DFB state. Additive recurrences are handled before this pass by
`ttl.accumulation_scope` lowering because strategy selection must choose
between DST residency and L1 packer accumulation. Generic bufferization
would emit memref load/store and would not produce either hardware
accumulation mechanism. The pass therefore implements iter_arg
elimination directly against DFB ops. It does not reuse bufferization's
conflict/aliasing analysis. Instead, it relies on the double-buffer invariant
described below.

### Lowering Strategies

Additive tensor recurrences use `ttl-form-accumulation-scopes` followed
by `ttl-lower-accumulation-scopes`. Formation records semantic accumulation
for recurrences that have a legal requested strategy. Lowering selects the
concrete storage strategy.

**DST strategy:** when the recurrence satisfies the DST legality rules,
the lowering creates a reduction-style `ttl.compute` with
`ttl.tile_accumulate ... add`. The generated `ttl.dst_section` spans the
reduction loop, so the accumulator stays in the destination register file
until the final store. This requires one acquire/release cycle around the
full reduction; releasing DST inside the reduction would lose the register
resident partial value.

**L1 packer strategy:** when DST is not selected, the lowering creates a
pre-loop non-accumulating store of the initial value and per-iteration
accumulating stores into the same output reservation. The loop is
annotated with `ttl.l1_acc_loop`, `ttl.l1_acc_initial =
accumulate_existing`, and `ttl.l1_acc_scope_id`. After TTKernel lowering,
`TTKernelInsertL1Accumulation` converts that metadata into
`pack_reconfig_l1_acc(1)` before the accumulating stores and
`pack_reconfig_l1_acc(0)` after them. With L1 accumulation enabled, the
packer adds each packed DST tile to the existing L1 tile instead of
overwriting it.

**General recurrence:** any tensor iter_arg that remains after
accumulation scope lowering uses compiler-allocated double-buffered DFB
state. The init is stored before the loop; each iteration consumes the
current state (`cb_wait`/`attach_cb`), computes, and produces the next
state (`cb_reserve`/`store`); a post-loop `cb_wait`/`attach_cb` yields
the final state value that replaces the loop result. This is not a
hardware accumulation optimization; it is the fallback representation for
recurrences that cannot be expressed as DST or L1 packer accumulation.

### Invariants

Preconditions:

- Runs on `func.func` nested in a `ModuleOp`, once per `scf.for`.
- Additive scope insertion matches only when all of the following hold;
  any tensor iter_arg failing them remains for general DFB state
  materialization:
  - the loop result has exactly one use, a non-accumulate `ttl.store`;
  - the yielded value is a single-use `ttl.add` in the loop body;
  - the iter_arg is one add operand and has no other use;
  - the other operand (the contribution) is not the iter_arg;
  - the store's destination is a `cb_reserve` whose only other uses are
    result-unused `attach_cb`s;
  - that `cb_reserve` and the store sit in the loop's parent block.

Postconditions:

- The rewritten `scf.for` carries no tensor iter_args or results;
  non-tensor iter_args keep their relative order.
- Additive scope lowering consumes every inserted `ttl.accumulation_scope`.
  `ttl-materialize-loop-state` then removes any remaining tensor iter_arg,
  so no tensor iter_arg reaches compute lowering.

Structural invariants:

- Each compiler-allocated state DFB is created with block count 2
  (`DFBMaterialization.cpp`), a fixed double buffer. The pass does not
  size it from the loop; it assumes one carried value in flight per
  iteration, so two slots suffice and larger counts would only waste L1.
  Its `bind_cb` is emitted at function entry, where `finalize-dfb-indices`
  requires compiler-allocated binds to live.
- The general strategy emits exactly one consume and one produce of the
  state DFB per iteration, keeping `cb_reserve`/`cb_wait` accounting
  balanced.
- Correctness assumes the loop-carried state is consumed before it is
  reproduced within an iteration, so two slots suffice. The pass does not
  verify this; it holds for the recurrences the frontend emits.

### Performance

DST-resident accumulation is the preferred mechanism when the compiler
can keep the recurrence inside one acquire/release cycle. The partial
value never leaves the register file, and the final result is packed
once.

L1 packer accumulation is used when the selected strategy materializes
the recurrence as stores to an output DFB across loop iterations. The
packer adds in place in L1 and avoids the DFB-to-DST load that an
explicit add would perform every iteration. Enabling L1 accumulation by
default in the d2m backend produced a significant measured speedup:
https://github.com/tenstorrent/tt-mlir/pull/8387.

General DFB state materialization does not use a hardware accumulation
mechanism. It round-trips the state through L1 each iteration because a
non-additive recurrence cannot be expressed as DST or packer
accumulation.

## Loop-Carried Tensor State

A Python `for` loop that reassigns a tensor variable read on a later
iteration (`acc = acc + x`, `state = ttl.math.relu(state)`) compiles to an
`scf.for` with a ranked-tensor `iter_arg`. `ttl-materialize-loop-state`
eliminates those tensor iter_args before compute lowering by creating
compiler-managed DFB state:

```
store init -> state DFB
for ...:
    wait/attach state DFB
    compute next state
    reserve/store next state -> state DFB
wait/attach final state DFB
```

The pass preserves non-tensor loop iter_args. It also preserves zero-trip
loop semantics because the initial value is stored before the rewritten loop
and the final value is read after the loop.

## DstSectionOp

`ttl.dst_section` demarcates a DST register acquisition scope. All
tile compute ops and stores in the body share one acquire/release
cycle. When lowered to TTKernel (`expandDstSections` in
`ConvertTTLToTTKernel`), the body is split at the first `TileStoreOp`
into math and pack phases:

    acquire -> [math ops] -> commit -> wait -> [pack ops] -> release

Three placement modes:

- **Non-subblocked**: one `dst_section` per tile loop iteration
- **Subblocked**: one `dst_section` wrapping the unrolled tile sequence
- **Accumulating**: one `dst_section` per parallel iteration, with
  the reduction loop inside

All computes use `DstSectionOp`, including matmul (`LowerMatmulBlock`).
Matmul K accumulation is currently selected from block sizes, DST capacity,
and subblocking behavior. #652 tracks an explicit
`matmul-k-accumulation=auto|dst|l1-pack` option so users and tests can require
DST-resident K accumulation or L1 packer accumulation directly.

## DST vs L1 Reduction Lowering

Two optimized lowerings exist for additive multi-tile reductions:

**DST accumulation** (`dst-accumulation=true`): Reorders loops so
parallel dims are outer and reduction dims are inner. `DstSectionOp`
wraps the reduction loop, so DST persists across iterations. One
pack after the entire reduction. More efficient (no L1 round-trip)
but holds the output DFB reserve longer.

**L1 accumulation** (`dst-accumulation=false`): Loops in declaration
order with per-tile `DstSectionOp`. Each iteration acquires DST, computes,
and packs. For overwrite-mode reductions, `pack_reconfig_l1_acc(1)` is
inserted after iteration 0 so later iterations add to the existing L1
value. See the "Reconfiguration placement around L1 accumulation loops"
section below for the full enable/disable sequence and the
`accumulate_existing` variant.

Selection: the `dst-accumulation` pass option on `ttl-lower-to-loops`
controls reduction lowering. The pipeline maps `maximize_dst` to this
option. This is separate from `accumulation-strategy`, which controls tensor
recurrence scopes.
`reduce_max` always uses DST accumulation because L1 accumulation
(`pack_reconfig_l1_acc`) accumulates via addition, which is only
correct for sum.

## Loop Structure

### DST Accumulation (parallel-outer, reduction-inner)

`generateAccumulatingLoops` separates parallel and reduction dims
from `iterator_types`:

```
for each parallel dim:           // output tile iteration
    dst_section {
        for each reduction dim:  // accumulate into DST
            <tile ops>
        <stores with placeholder tile + explicit dst_index>
    }
```

Stores use a placeholder tile value (via `UnrealizedConversionCastOp`)
with an explicit `dst_index` operand, since the SSA tile value from
`reduce_tile` is loop-local.

### L1 Accumulation (declaration-order loops)

```
for each dim (declaration order):
    dst_section {
        <tile ops>
        <stores>
    }
```

Reduction loops are annotated with `ttl.reduction_loop`,
`ttl.l1_acc_initial`, and `ttl.l1_acc_scope_id`.
`TTKernelInsertL1Accumulation` consumes that metadata after conversion to
place packer L1 accumulation reconfiguration.

### Reconfiguration Placement Around L1 Accumulation Loops

`TTKernelInsertL1Accumulation` brackets each semantic scope group with
`pack_reconfig_l1_acc` calls. Scope groups are formed from
`ttl.l1_acc_scope_id`; the pass no longer infers semantic grouping from
shared pack dataflow buffers. The standard overwrite sequence disables
L1 accumulation before the group, conditionally enables it after the
first iteration's last pack so subsequent iterations accumulate, and
disables it again after the group:

```
pack_reconfig_l1_acc(0)
for iv = lb..ub:
    ...pack...
    if iv == lb: pack_reconfig_l1_acc(1)
pack_reconfig_l1_acc(0)
```

If a later non-group pack targets one of the group's output dataflow buffers
before the corresponding push, the disable is emitted before that pack. Packer
L1 accumulation state is global, so the later pack must not observe the enabled
state.

If a pack inside an annotated loop targets a DFB that is not published as an
accumulated scope output, the pass emits a temporary disable before that pack
and restores the active scope state after it. Format validation applies only to
packs that target the accumulated output DFBs.

When `ttl.l1_acc_initial = accumulate_existing`, lowering has already
proved that L1 holds the initial value for the scope. The reconfiguration
before the group enables L1 accumulation, and the per-iteration
conditional enable on the root loop is omitted because every iteration
must accumulate from iteration 0 onward:

```
pack_tile(...)                  // prior pack runs with L1 accumulation disabled
pack_reconfig_l1_acc(1)
for iv = lb..ub:
    ...pack...
pack_reconfig_l1_acc(0)
```

The loop producer selects between the two sequences with
`ttl.l1_acc_initial`. `overwrite` disables L1 accumulation before the loop so
iteration 0 writes the baseline tile. `accumulate_existing` enables L1
accumulation before the loop so iteration 0 adds onto a value materialized
by an earlier store.

The pass is idempotent: a prior run leaves a `pack_reconfig_l1_acc`
either inside the L1 accumulation loop body or immediately preceding the loop,
and the second run detects either signal and returns.

## Per-Op Init Insertion

`TTKernelInsertInits` uses two targeted walks instead of a block walk:

1. `walk(TileRegsAcquireOp)`: iterates top-level ops between acquire and
   release. Each top-level op may contain compute ops in nested regions
   (e.g., `reduce_tile` inside a reduction `scf.for`); these are
   discovered via `op.walk()`. Init is inserted before the top-level
   container op. Consecutive ops with the same init configuration share one
   init by tracking the previous configuration while walking forward.

2. `walk(func::FuncOp)`: handles compute ops outside sync regions
   (unit tests). Skips ops already processed by walk 1.

Broadcast, reduce, and transpose inits resolve their output DFB from a
`ttl.*_output_cb_index` attribute propagated during TTL-to-TTKernel
conversion.

## External References

Hardware statements about DST registers, L1 packer accumulation, and
`fp32_dest_acc_en` are based on these commit-pinned tt-metal references:

- [Compute engines and dataflow within Tensix](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst).
- [Matrix engine technical report](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/tech_reports/matrix_engine/matrix_engine.md).
- [L1 accumulation FP32 analysis](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/docs/L1_ACCUMULATION_FP32_ANALYSIS.md).

Handwritten-kernel observations are based on internal kernel audits and
benchmark notes.

## IR Trace: 2x2 reduce_sum along dim 0

Input: `tensor<2x2xtile>`, scaler: `tensor<1x1xtile>`,
output: `tensor<1x2xtile>`.

### DST Accumulation (dst-accumulation=true)

After LowerToLoops:
```mlir
scf.for %j = %c0 to %c2 step %c1 {       // parallel
    ttl.dst_section {
        scf.for %i = %c0 to %c2 step %c1 { // reduction
            %in = tensor.extract %inp[%i, %j]
            %sc = tensor.extract %scaler[%c0, %c0]
            %out = tensor.extract %init[%c0, %j]
            ttl.tile_reduce %in, %sc, %out sum reduce_dim_col into dst[%c0]
        } {ttl.reduction_loop, ttl.tile_loop_stride = 2}
        ttl.tile_store %placeholder, %view[%c0, %j] from dst[%c0]
    }
} {ttl.tile_loop_stride = 1}
```

After TTKernel conversion + insert-inits:
```
init_sfpu(cb0, cb2)
for j = 0..2:                              // parallel
    tile_regs_acquire()
    reduce_init(cb0, cb1, cb2, SUM, REDUCE_COL)
    for i = 0..2:                          // reduction (DST persists)
        reduce_tile(cb0, cb1, i*2+j, 0, 0, SUM, REDUCE_COL)
    reduce_uninit()
    tile_regs_commit() / tile_regs_wait()
    pack_tile(0, cb2, j)
    tile_regs_release()
cb_push_back(cb2, 2)
```

### L1 Accumulation (dst-accumulation=false)

After LowerToLoops:
```mlir
scf.for %i = %c0 to %c2 step %c1 {       // reduction (declaration order)
    scf.for %j = %c0 to %c2 step %c1 {   // parallel
        ttl.dst_section {
            ttl.tile_reduce ... into dst[%c0]
            ttl.tile_store ...
        }
    } {ttl.tile_loop_stride = 1}
} {ttl.reduction_loop, ttl.tile_loop_stride = 2}
```

After TTKernel conversion + insert-inits + L1 accumulation:
```
init_sfpu(cb0, cb2)
pack_reconfig_l1_acc(0)
for i = 0..2:                              // reduction
    for j = 0..2:                          // parallel
        tile_regs_acquire()
        reduce_init(...)
        reduce_tile(cb0, cb1, i*2+j, 0, 0, SUM, REDUCE_COL)
        reduce_uninit()
        tile_regs_commit() / tile_regs_wait()
        pack_tile(0, cb2, j)               // overwrites or adds to L1
        tile_regs_release()
    if (i == 0) pack_reconfig_l1_acc(1)
cb_push_back(cb2, 2)
pack_reconfig_l1_acc(0)
```
