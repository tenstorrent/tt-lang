# `ComputeOp` Creation and Fusion

## Purpose

`ComputeOp` creation converts tensor operations and their output stores into
`ttl.compute` operations containing tile-level recipes. Fusion absorbs a
supported tensor expression into one `ttl.compute` so intermediate values can
remain in DST instead of being stored to dataflow buffers (DFBs).

Creating a `ComputeOp` may change both evaluation position and output
publication. A tensor operation may be defined before a DFB release, used in
another region, stored through several reserves, or shared by several
consumers. Correctness therefore requires more than following SSA definitions.
The compiler must prove:

- every input DFB remains available where the `ttl.compute` executes;
- the compute preserves each output reserve, store, and push transaction;
- the replacement result dominates every surviving use;
- overlapping fusion candidates are applied in an order that preserves their
  recorded source operations;
- instrumentation retains its observable ordering; and
- every `ttl.store` is assigned to a creation plan or a DFB-to-DFB passthrough
  before conversion modifies IR.

DFB acquisition identity, release ownership, and program-point availability
are defined in [DFBManagement.md](DFBManagement.md).

## Design Motivation

`ComputeOp` creation is a nonlocal transformation. Candidate legality depends
on operations, uses, publication transactions, and DFB lifetimes throughout a
kernel. Rewriting one candidate changes use lists, operation positions, and
transactions that another candidate might inspect. Greedy rewrite order
therefore cannot define semantic decisions without making legality depend on
earlier mutations.

The implementation follows two established upstream analyze-then-apply
designs. [One-Shot Bufferize](https://github.com/llvm/llvm-project/blob/4279d524cc78d0bac294bb29257c62665121d9f1/mlir/include/mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h)
records aliasing and in-place decisions in analysis state before rewriting IR.
[LLVM VPlan](https://github.com/llvm/llvm-project/blob/4279d524cc78d0bac294bb29257c62665121d9f1/llvm/lib/Transforms/Vectorize/VPlan.h)
represents selected recipes and dependencies before generating vectorized IR.
TT-Lang similarly builds and validates a complete immutable kernel plan before
the first mutation. Application executes only recorded decisions, and no plan
is reused after an IR mutation. Upstream dataflow, dominance, region-order, and
purity analyses provide general compiler facts; TT-Lang adds only DFB protocol
and tile-recipe semantics.

## Analysis and Mutation Separation

The former conversion interleaved analysis and mutation:

```text
repeat in greedy rewrite order:
  inspect the current source operation and use lists
  trace a fusable expression to its current DFB-backed inputs
  infer iteration, output DFBs, and instrumentation placement
  create ttl.compute and immediately erase or move source operations
```

An earlier rewrite could therefore change the inputs, stores, pushes, uses, or
instrumentation consulted by a later decision. Overlapping fusion candidates
also had no kernel-wide ownership or application order.

The current conversion separates those responsibilities:

```text
analyze one immutable kernel:
  compute DFB value availability at every relevant program point
  construct complete typed direct, fused, and elision candidates
  record reserve/store/push output transactions
  validate input lifetimes, result dominance, and instrumentation placement
  select non-conflicting candidates and order overlapping candidates
  assign every store or report why it remains unassigned

apply the validated kernel plan:
  verify that recorded operands, uses, and transactions still match
  create only the recorded ttl.compute recipes in the recorded order
  erase only the recorded stores and absorbed operations
```

Intermediate DFB insertion uses the same separation. It computes a monotone
fixed point of exact consumer operands requiring storage, groups those
requirements by producer, and applies the complete materialization plan only
after analysis terminates. The final conversion analyzes the modified kernel
again; plans are never reused after mutation.

## Terminology

A **source operation** is a tensor operation considered for replacement by a
`ttl.compute`.

A **direct creation** replaces one source operation with a `ttl.compute` that
contains its operation-specific tile recipe.

A **fused creation** traces from a source result through supported producer
operations and emits their tile recipes in one `ttl.compute` body. The plan
**absorbs** a producer when creating the `ttl.compute` erases that producer.

A **root input** is an external tensor input to a fused expression. It is
either already DFB-backed or selected for compiler-DFB materialization.

A **lifetime root input** is a root whose current storage remains an input
after all planned materializations. Only lifetime roots constrain the
`ComputeOp` insertion point through the DFB availability analysis.

An **output publication transaction** contains one `cb_reserve`, every
`ttl.store` using a view derived from that reserve, and the matching `cb_push`
when one is present.

An **elision** replaces an operation without creating a `ttl.compute` or
changing its evaluation position. Identity `ttl.typecast` is the current
elision.

## Pipeline Contract

`ComputeOp` creation and intermediate materialization use this order:

```text
ttl-create-producer-compute
  -> ttl-insert-intermediate-dfbs
  -> convert-ttl-to-compute
  -> ttl-auto-sync
```

`ttl-create-producer-compute` applies creation plans that are legal in the
original kernel. Rejected candidates remain unchanged so intermediate DFB insertion can
repair storage or publication constraints. `ttl-insert-intermediate-dfbs`
computes its own immutable plan, inserts the required storage, and rewrites the
selected consumer operands. `convert-ttl-to-compute` then rebuilds creation
and lifetime plans from the modified kernel and requires every output store to
be assigned. `ttl-auto-sync` inserts absent DFB pushes and pops after final
uses are known.

No plan survives a pass that modifies the kernel. Each of the three TT-Lang
pipeline entry points must preserve and test this order:

- `lib/Dialect/TTL/Pipelines/TTLPipelines.cpp`, the registered C++ pipeline;
- `python/ttl/ttl_api.py`, the runtime Python compiler pipeline; and
- `test/me2e/builder/pipeline.py`, the ME2E entry point that invokes the
  registered pipeline.

## Design Structure

The implementation has four responsibilities:

```text
DFB value lifetime analysis
  -> candidate planning and legality
  -> kernel-wide candidate selection and application order
  -> mechanical pattern application
```

`DFBValueLifetimeAnalysis` answers whether a DFB-backed tensor is definitely
available before an operation. `ComputeOpCreationPlanner` builds typed
candidate plans and selects a dependency-safe kernel plan.
`ConvertTTLToCompute.cpp` contains application patterns that consume those
plans. `IntermediateDFBPlanner` determines which operand values must receive
new storage before final creation.

The planners do not modify IR. Operation handles and SSA values in a plan are
valid only while the analyzed kernel is unchanged.

## Compute Target Capabilities

`ttcore::TileType` validates dimensions that `tt_metal::Tile` can represent.
TTL operation verifiers enforce target-independent type relations, including
matmul K compatibility and result dimensions. Neither layer defines LLK
capabilities.

`ComputeTargetEnvironment` owns architecture-specific compute constraints. It
validates kernel tile dimensions, primitive data-type support, and matmul LLK
combinations before compute creation or TTKernel conversion modifies IR. The
environment is selected from the module's system description or
`ttl.target_arch`. Modules without either use the intersection of capabilities
implemented by all supported targets. Wormhole B0 and Blackhole currently
provide the same compute capability set.

Adding an architecture requires a capability implementation and an entry in
`computeTargetRegistrations`. The registration table is the source for both
explicit target lookup and the common environment, so an architecture cannot
be available explicitly without contributing to the common intersection. An
architecture may reuse an implementation only when the shared contract is
documented. Tests must cover its accepted and rejected tile dimensions, data
types, and operation-specific combinations.

## Candidate Planning

Candidate planning records the source operands and result uses before
selecting a creation strategy:

```text
planCandidate(source):
  require one ranked-tensor result
  record source operands and every result use

  if source is an identity typecast:
    kind = position-preserving elision
  else if source has a direct recipe and its required inputs are DFB-backed:
    kind = direct creation
  else:
    trace source result through supported fusable producers
    require a non-empty successful trace
    kind = fused creation

  derive typed affine maps and iterator kinds
  plan output publication transactions
  validate instrumentation placement
  find SSA boundaries needed to preserve instrumentation order
  validate DFB input availability at the publication anchor
  validate dominance of every surviving result use

  return the complete plan and its typed legality result
```

The planner distinguishes three outcomes with `PlanningResult`:

- **planned**: the complete candidate is legal and may be applied;
- **rejected**: valid IR cannot use this creation, but the candidate or its
  reason may be required by dependency and materialization analysis; and
- **invalid IR**: the source violates a compiler precondition or a DFB
  transaction invariant and compilation must fail.

This distinction prevents an unsupported optimization from becoming an error
while ensuring malformed IR cannot continue into partial conversion.

### Direct Recipes

Direct recipes define the complete iteration and tile-level implementation for
one operation. Current recipes include elementwise operations, block
broadcast, matmul, reduce, multiply by a scalar constant, fill, typecast, and
transpose.

The plan stores typed `utils::IteratorType` values and affine maps. String
attributes required by the current `ttl.compute` syntax are created only when
the operation is built. Each recipe also records its operation-specific
properties, including reduction kind and dimension, matmul transposition,
broadcast kind, and scalar constants.

### Fusion Trace

Fusion tracing proceeds from a candidate result toward external inputs:

```text
trace(value):
  if value is DFB-backed:
    add value as a root and lifetime root
    return

  if the incoming operand is selected for materialization:
    add value as a root, but not as a lifetime root
    return

  if value is produced by a supported elementwise operation:
    trace every tensor operand
  else if value is produced by block broadcast or matmul:
    require each declared DFB input to be backed or materialized
    add the operation and its roots
  else if value is produced by fill:
    add the operation without roots
  else:
    reject the trace and identify the failed operand

  append the producer after its dependencies
```

`rootInputs` defines the eventual `ttl.compute` inputs. A materialized operand
remains in this set because its replacement DFB becomes a compute input.
`lifetimeRootInputs` contains only roots whose existing storage will still be
read. If one SSA value occurs through both materialized and unmaterialized
operands, the unmaterialized occurrence retains it in `lifetimeRootInputs`.
This distinction prevents the old storage lifetime from forcing redundant
materialization while preserving every remaining read.

### Fused Iteration Semantics

Each root use has an indexing role. Elementwise and broadcast uses are
parallel. Matmul left, right, and transposed-right uses have distinct
contraction roles. A root used in multiple roles appears in the created
`ttl.compute` input list once per distinct role so each occurrence receives the
correct affine map.

A fusion containing matmul uses an `[M, N, K]` iteration domain. Elementwise
inputs and results use `[M, N]`; matmul inputs use `[M, K]`, `[K, N]`, or
`[N, K]`. Other fused expressions use the result rank with broadcast-aware
parallel maps. The planner validates that every static tensor dimension agrees
with the iteration dimension selected by its affine map.

The tile recipes are recorded in dependency order. Elementwise operations,
fill, block broadcast, and matmul each have explicit recipe records. A matmul
feeding one add operand may use the hardware accumulator form when the other
operand provides a concrete accumulator tile. Two matmul results feeding the
same add remain separate because neither deferred result could initialize the
accumulator. Instrumentation between a matmul and add also keeps separate tile
operations and emits a warning because the combined hardware operation cannot
preserve the observation point.

### Cross-Region Recomputation

A pure producer defined in a dominating block may be recomputed in a nested
consumer block. SSA dominance proves its operands are available in that block.
Upstream `isPure` proves the operation is speculatable and has no memory
effects. DFB lifetime analysis independently proves that its storage inputs
remain available at the new execution position.

This permits fusion across `scf.if` and `scf.for` boundaries without treating
every region boundary as a storage requirement. Fusion is rejected when
instrumentation around a cross-block producer cannot be placed relative to the
consumer without changing observations.

## Output Publication Planning

Creation replaces tensor stores with tile stores inside `ttl.compute` and may
move an existing push after the new compute. Publication planning therefore
records reserve-delimited transactions before mutation:

```text
planPublications(source):
  collect every ttl.store of the source result
  require all stores to be in one block

  for each store:
    trace the destination view to one cb_reserve
    group stores by reserve identity
    find the first matching cb_push after the stores and before another
      reserve of the same DFB

  require all stores in one transaction to precede the same push
  insertion anchor = final source-result store
  record whether one DFB has more than one reserve transaction
```

Several reserves of different DFBs may become outputs of one compute. Several
reserves of the same DFB cannot be combined because moving one publication to
the final store would cross a later reserve of the same producer pointer. That
case requires intermediate materialization so each original transaction can
remain independent.

The final store is the insertion anchor. Every output reserve dominates its
store, so all output views are available at that position. The planner rejects
creation unless each lifetime root is definitely available there and the
anchor dominates every result use not removed by the creation.

Existing pushes are resolved again during application because an earlier
creation may relocate a shared push. The reserve and store identities remain
the analyzed ones; this limited resolution does not repeat semantic analysis.

## Instrumentation

`ttl.signpost` and tile-observing debug prints are not ignored during fusion.
The plan records each movable observation relative to an absorbed operation or
output store. A complete signpost scope may remain around the created
`ttl.compute`.
A partially overlapping or unmatched scope rejects fusion.

Instrumentation belongs to the creation that erases its source operation. If
another creation recomputes that source, the recomputed tile recipe does not
duplicate the instrumentation. This preserves one observable event sequence
around the operation that replaces the original source.

For each fused creation, the planner classifies every operation between the
first tensor operation moved into the `ttl.compute` body and the final output
store using MLIR's `isPure`. An operation that is not both
speculatable and memory-effect-free cannot be reordered with instrumentation.
When a recorded instrumentation operation would move from before such an
operation to after it, the planner records the exact tensor SSA uses crossing
the same boundary. Intermediate DFB planning materializes those uses, so each
side receives an independent `ttl.compute`, and the instrumentation retains its
original order.
Pure tensor recomputation without movable instrumentation requires no split.
Output reserves are excluded because they supply the formal output views and
the compute must execute after them. `ttl.dprint` declares memory effects;
scalar, DFB, and tensor prints stay outside the compute, while tile and DST
prints use the explicit relocatable instrumentation plan.

Instrumentation outside the sink block cannot be ordered relative to a
cross-block recomputation and therefore prevents that fusion. These rules are
conservative: they may retain separate computes, but they do not silently
change the observed operation order.

## Kernel-Wide Candidate Selection

Candidate legality alone does not determine application order. Fused
expressions may overlap, and applying one creation may erase a source needed
by another. The kernel planner computes the erase set of every legal fused
candidate using the same reverse use-empty rule as application.

For example:

```mlir
%sum = ttl.add %lhs, %rhs : ...
ttl.store %sum, %published : ...
%first = ttl.exp %sum : ...
ttl.store %first, %first_output : ...
%second = ttl.exp %sum : ...
ttl.store %second, %second_output : ...
```

Both `ttl.exp` creation plans may absorb `ttl.add`, but `%sum` also requires an
independent creation plan for its publication. The absorbing creations execute
before the independent `%sum` creation. Each earlier creation removes only its
own consumer; another use keeps `ttl.add` present until all absorbers have run.

The selection algorithm is:

```text
buildKernelPlan(kernel):
  build a complete candidate record for every one-result source

  repeat:
    accept a result-use-dominance rejection when every offending use is
      removed by an already legal fused creation
  until no candidate changes

  for each rejected candidate requiring intermediate DFB repair:
    mark its absorbed candidate sources as deferred

  for each legal candidate:
    compute which absorbed sources it erases
    omit an absorbed candidate erased by another selected creation
    otherwise require the absorber to execute before the absorbed source

  order identity elisions from SSA definitions to users
  topologically schedule the remaining creation dependencies
  record one surviving use for each source retained after an earlier absorber

  build DFB-to-DFB passthrough plans for remaining stores
  record every store not assigned to a selected plan
```

Only rejection kinds that intermediate DFB insertion can repair defer an
absorbed producer: an unmaterialized input, repeated output transaction,
released input, or non-dominating result use. An unsupported recipe or output
form creates no such dependency. This prevents a permanently unsupported
consumer from suppressing an independently legal producer.

The plan records stable use identities as an owner operation and operand
number. It also records uses that preceding creations must remove and one use
that must preserve a later source. Application verifies these facts before
rewriting, so a stale or inconsistent plan fails without partially applying a
different creation.

## Intermediate DFB Interaction

Intermediate DFB analysis follows the One-Shot Bufferize decision model. It
records exact consumer operands that require new storage and grows that set to
a fixed point before mutation. A creation trace stops at those operands and
treats each as a future DFB-backed root.

The lifetime-root distinction is required when the selected operand is already
DFB-backed but may be released. For example:

```mlir
%identity = ttl.typecast %input : (tensor<...xbf16>) -> tensor<...xbf16>
ttl.cb_pop %input_dfb : ...
%reduced = ttl.reduce %identity, %scaler ...
```

Producer creation elides the identity typecast at its original position. The
reduce then reads `%input` directly after its DFB release. Intermediate DFB
analysis records the reduce operand with the reason
`dfb-input-may-be-released` and materializes `%input` before the pop. The
replacement compiler DFB is a root of the eventual reduce creation plan, but the
released input DFB is no longer a lifetime root. The reduce result therefore
does not receive a redundant materialization.

This rule is not specific to typecast or reduce. Every
`DFBInputOpInterface` operand is materialized when it is unattached or when its
attached storage may be released before the consumer. With compiler DFBs
disabled, either condition produces a diagnostic instead of changing the
lifetime result.

## Plan Application

`ttl-create-producer-compute` and `convert-ttl-to-compute` use the same planner
and application patterns with different completeness policies. Producer
creation permits unassigned stores that intermediate DFB insertion may
repair. Final conversion diagnoses the first unassigned store before any
rewrite.

Application processes only operations present when planning completed:

```text
applyKernelPlan(plan):
  reject an operand, result use, preserving use, or instrumentation mismatch

  for source in creation order:
    select the recorded direct, fused, or elision pattern
    resolve current pushes for recorded output transactions
    build ttl.compute from recorded inputs, maps, iterators, and tile recipes
    replace recorded result uses
    erase only absorbed operations whose uses are now empty

  apply recorded passthrough-store conversions
```

Greedy folding is disabled during ordered application. Patterns cannot infer a
new recipe, change the input set, or bypass a rejection. A selected source that
remains after its pattern executes is an internal compiler error. This makes
the rewrite stage mechanical and keeps policy in the immutable planner.

## Correctness Argument

The design preserves these properties:

1. **Tensor value.** Each direct or fused recipe records the tile operation,
   operand relation, result type, affine maps, and iterator kinds needed to
   implement the source expression. Static iteration extents are checked for
   consistency before application.

2. **Input storage.** Every current-storage input is definitely available at
   the `ComputeOp` insertion point. A planned materialization replaces its exact consumer
   operand and supplies storage whose pop is inserted after the final use.

3. **Control flow.** Cross-block recomputation is restricted to pure
   operations whose SSA operands dominate the consumer. DFB availability is
   evaluated at the consumer position through MLIR control-flow dataflow.

4. **Publication order.** Each tile store retains its reserve transaction.
   Moving a push cannot cross another reserve of the same DFB because repeated
   transactions of one DFB reject combined creation.

5. **SSA uses.** The insertion anchor dominates every surviving result use.
   Uses that a preceding creation must erase are recorded and verified.

6. **Instrumentation order.** Instrumentation does not move across an
   operation unless MLIR proves that operation pure. Intermediate DFB
   materialization splits the tensor SSA frontier when required. Output
   reserves are recorded dependencies of the created `ttl.compute` and therefore
   remain ordered before it.

7. **Overlapping candidates.** An absorbed source is either erased by one
   selected creation or retained until every preceding absorber executes. A
   recorded surviving use proves the latter condition during application.

8. **Analysis stability.** No mutation occurs until the whole kernel plan is
   valid. Application checks the source operands, result uses,
   instrumentation, and output transactions used by the proof.

9. **Conversion completeness.** Final conversion assigns every `ttl.store` to
   a selected `ComputeOp` creation or passthrough plan before modifying IR.

The proof assumes verified TTL operation types, valid DFB FIFO semantics, and
recognized view-preserving operations for acquired DFB storage. Conservative
or unresolved DFB ownership returns "may be released" and may require a
compiler DFB; it never proves an unsafe creation legal.

## Upstream MLIR and LLVM Reuse

The implementation reuses upstream infrastructure for general compiler
relations:

- MLIR dense forward dataflow and dead-code analysis for executable
  program-point facts;
- `DominanceInfo` for SSA availability across blocks and regions;
- MLIR region-aware topological sorting for materialized compute producers;
- memory-effect and speculation interfaces through `isPure` for cross-region
  recomputation;
- `TilingInterface`, `DestinationStyleOpInterface`, and
  `IndexingMapOpInterface` to expose the created `ttl.compute` operation's
  planned iteration semantics to later transformations;
- greedy pattern infrastructure for mechanical application after planning;
- One-Shot Bufferize's analysis-state model for monotone operand decisions;
  and
- LLVM VPlan's immutable-plan/application separation as the model for
  dependency-ordered transformation.

The corresponding upstream implementations are:

- [dense forward dataflow](https://github.com/llvm/llvm-project/blob/4279d524cc78d0bac294bb29257c62665121d9f1/mlir/include/mlir/Analysis/DataFlow/DenseAnalysis.h)
- [dead-code analysis](https://github.com/llvm/llvm-project/blob/4279d524cc78d0bac294bb29257c62665121d9f1/mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h)
- [One-Shot Bufferize analysis state](https://github.com/llvm/llvm-project/blob/4279d524cc78d0bac294bb29257c62665121d9f1/mlir/include/mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h)
- [LLVM VPlan](https://github.com/llvm/llvm-project/blob/4279d524cc78d0bac294bb29257c62665121d9f1/llvm/lib/Transforms/Vectorize/VPlan.h)

Upstream does not model the TTL-specific relations required here. TT-Lang adds:

- DFB FIFO acquisition identity and conservative release ownership;
- DFB value availability at a creation position;
- reserve/store/push publication transactions;
- direct and fused TTL tile recipes with hardware-specific affine roles;
- overlapping-candidate selection and preservation records;
- three-state planned, rejected, and invalid-IR results; and
- evidence-bearing intermediate DFB requirements.

The split keeps CFG, dominance, purity, and structured-operation mechanics in
upstream infrastructure while limiting TT-Lang code to DFB protocol and TTL
code-generation semantics.

## Diagnostics and Testing

`ttl-print-compute-op-creation-plans` prints candidate kinds, recipes, iteration
semantics, legality reasons, application order, unassigned stores, and each
intermediate DFB requirement with its evidence. IDs use kernel walk order, so
the output is deterministic and suitable for regression tests. The pass does
not modify IR.

The compiler tests cover direct and fused recipes, overlapping candidates,
cross-region recomputation, instrumentation placement, output transactions,
released inputs, multi-output accumulating computes, plan invalidation, and
disabled compiler DFBs. Runtime tests validate representative creation and
materialization results in bf16 and f32. The largest representative program is
the eight-node flash-attention chain, which combines many atom-composed tensor
operations and user DFB publications.

## Limitations and Future Work

- Output stores for one source operation must share a block. Conditional
  routing to different output DFBs is diagnosed instead of asserting, but it
  does not yet create a `ttl.compute`.
- Tensor-operation candidates currently require one result. Existing
  multi-output `ttl.compute` operations are supported by materialization and
  loop lowering.
- Fused expressions require explicit TTL tile recipes. Unsupported tensor
  operations stop tracing and remain separate or require materialization.
- Cross-block fusion rejects instrumentation that cannot be ordered relative
  to the sink block.
- `ttl.tile_store` does not encode its formal output index. The compiler traces
  its view to a DFB and requires formal outputs to use distinct DFBs. Issue
  #797 tracks replacing this temporary association.
- Entry-block DFB release ownership uses exact static FIFO matching; ambiguous
  nested control flow remains conservative. Issue #724 tracks stronger
  structured-control-flow balance reasoning.
- Candidate selection is legality-based. A future cost model may compare legal
  recomputation and materialization strategies without changing the analysis
  or application contracts.

## Implementation Files

- `lib/Dialect/TTL/Transforms/ComputeOpCreationPlanning.{h,cpp}` defines typed
  recipes, publication plans, legality, overlap handling, and kernel order.
- `lib/Dialect/TTL/IR/TTLOpsUtils.cpp` implements fusion tracing.
- `lib/Dialect/TTL/Transforms/DFBValueLifetimeAnalysis.{h,cpp}` provides
  program-point storage availability.
- `lib/Dialect/TTL/Transforms/IntermediateDFBPlanning.{h,cpp}` computes the
  fixed point of storage requirements.
- `lib/Dialect/TTL/Transforms/ConvertTTLToCompute.cpp` mechanically applies
  direct, fused, elision, and passthrough plans.
- `lib/Dialect/TTL/Transforms/TTLInsertIntermediateDFBs.cpp` applies grouped
  compiler-DFB materialization plans.
- `lib/Dialect/TTL/Transforms/TTLPrintComputeOpCreationPlans.cpp` prints
  deterministic planner state for debugging and tests.
