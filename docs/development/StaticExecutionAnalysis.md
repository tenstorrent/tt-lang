# Static Execution Analysis

This document defines the compile-time analysis used to count operation
executions within structured control flow. The PipeNet communication-protocol
verifier is the first consumer. A compiler cost model can combine these counts
with latency and concurrency information to assign comparable estimates to
legal IR alternatives.

## Semantics

A static operation is one operation in the compiler intermediate
representation (IR). A dynamic instance is one runtime execution of that
static operation. An IR region contains blocks and operations. Structured
control flow represents loops and conditionals as operations with child
regions.

Each query selects a root region, which the analysis assumes executes once. An
analysis consumer is a compiler component that queries the analysis. An
analysis context is the set of additional compile-time facts supplied by that
consumer, such as launch coordinates.

The analysis answers this question:

> How many dynamic instances of an operation execute during one invocation of
> a specified root region?

For a kernel function, the function body is the root region and the result is
relative to one function invocation. To compute a count across function calls,
a consumer multiplies the per-invocation count by each call operation's dynamic
instance count and sums the contributions from all calls to the function.

The result is either:

- An exact unsigned 64-bit count, including zero.
- Unknown because an exact count is not proven.

Unknown is distinct from zero and one. A correctness transformation that
requires an exact count must reject an unknown result. A consumer estimating
performance may preserve the unknown value or use another analysis that
represents unknown values symbolically or as numeric bounds.

An exact loop iteration count is necessary but not sufficient for operations
inside the loop. The analysis must also prove how often the nested regions and
blocks execute. A count is unknown when it depends on runtime data that the
analysis context cannot evaluate, such as a function argument, a memory load,
or a value modified by successive loop iterations without a proven constant.

The count describes dynamic operation instances. It does not describe their
execution order, latency, parallelism, or overlap. For example, a parallel loop
with 64 iterations contributes 64 instances, but this analysis does not assert
whether those instances execute concurrently.

## Analysis context

Some compile-time values are known only to a consumer. The analysis accepts two
callbacks:

- A symbol-value evaluator returns an integer for a static single assignment
  (SSA) value whose compile-time value is known to the consumer but is not
  encoded as an IR constant. Its result must have the same bit width as the SSA
  value's integer type. PipeNet analysis uses it for launch coordinates and
  conditions that select source and destination nodes.
- A region-invocation evaluator returns an exact invocation count for a
  non-loop child region whose semantics are known to the consumer. PipeNet
  analysis uses it for coordinate-dependent `ttl.if_src` and `ttl.if_dst`
  regions and for the unconditional `ttl.pipenet_scope` region.

A loop induction variable is the SSA value that identifies the current
iteration. The shared integer-expression evaluator combines constants, loop
induction variables, and consumer facts, then uses each operation's standard
folding semantics. Consumers do not duplicate integer operation rules.

Integer evaluation uses an explicit worklist and a memo table for each stable
induction environment. Each value in a shared expression graph is evaluated at
most once, and expression depth does not consume the C++ call stack.

Callbacks return facts, not defaults. If a callback supplies no fact, the
analysis uses the standard Multi-Level Intermediate Representation (MLIR)
interfaces when possible. The result remains unknown when neither mechanism
proves an exact value or count.

## Control-flow representation

A block invocation is one execution of a block during one invocation of its
containing region. The block invocation count is relative to one region
invocation. The root block is the block in the root region that contains the
outermost operation relevant to the query.

A control frame contains a parent operation, one child region of that parent,
and the block in that child region that contains the next nested operation. Its
region invocation count is the number of times the parent invokes that child
region. Its block invocation count is the number of times its block executes
per child-region invocation. For an operation nested under structured control
flow, the analysis records an ordered control-frame sequence from the root
region to the operation. This document names that sequence `control_frames`.

A control-frame sequence is induction-independent when no region or block
invocation count depends on an induction variable from an earlier loop frame.
Its operation count is:

```text
operation_count =
    block_invocation_count(root_block)
    * product(region_invocation_count(frame)
              * block_invocation_count(frame.block)
              for frame in control_frames)
```

The analysis evaluates factors from outermost to innermost. It returns unknown
when the next factor is not exact. An exact zero reached before that point ends
the proof because the nested operation is unreachable in that analysis context.

An induction environment maps each enclosing loop induction variable to the
value for one statically evaluated iteration. A loop is statically enumerable
when the analysis can compute every induction value and their number does not
exceed the configured enumeration limit. For one loop frame, let
`induction_values` be the computed sequence of values for that loop's induction
variable, and let `nested_count(induction_value)` be the count from the
remaining frames under one such value. When a later frame's invocation count
depends on this loop's induction variable, the operation count is:

```text
operation_count =
    sum(nested_count(induction_value)
        for induction_value in induction_values)
```

A rectangular loop nest is one in which no loop trip count depends on an outer
loop induction variable. The product above avoids enumerating large rectangular
loop nests. Enumeration remains available for predicates such as
`if iteration < static_limit` when `static_limit` is known at compile time.

## MLIR interfaces

The analysis uses MLIR interfaces as the primary semantic source:

- `LoopLikeOpInterface::getStaticTripCount()` supplies an exact loop trip
  count, meaning the number of loop iterations, when the loop operation can
  compute one. The analysis requires one loop region because the interface does
  not define separate invocation counts for regions in a multi-region loop.
- `RegionBranchOpInterface::getRegionInvocationBounds()` supplies region
  invocation bounds: the minimum and maximum number of times the parent may
  invoke each child region. A region count is exact only when its lower and
  upper bounds are equal.
- `BranchOpInterface::getSuccessorForOperands()` identifies a unique block
  successor when the branch operands are compile-time constants. Otherwise,
  the analysis retains every successor that may execute.

Constant operands are passed to `getRegionInvocationBounds()`. Integer
operands that become constant in the current analysis context are also passed
as MLIR attributes, which represent compile-time values in the IR.

An entry region is a child region that the parent operation may invoke first. A
selector is an operand whose value determines which entry region is selected.
The region-successor graph contains the child regions as nodes and possible
transitions between regions as edges. The target region is the child region in
the control frame currently being counted.

Some operations that implement `RegionBranchOpInterface` report an invocation
range of zero or one for each entry region even when a compile-time-known
selector identifies the region that executes. The analysis combines those
bounds with `getEntrySuccessorRegions()` and the region-successor graph. A
selected entry region executes at least once because it is the unique entry
successor. Its count is exactly one when either its upper bound is one or every
successor returns to the parent operation. This proves the exact count without
operation-specific condition handling. The analysis does not otherwise treat
an upper bound as an exact count because the lower bound may be smaller.

When a compile-time-known selector identifies an entry region other than the
target, the analysis walks the region-successor graph. The target region has
count zero only when no successor sequence from the selected region can reach
it.

`scf.for` receives additional support when the analysis context can evaluate
its lower bound, upper bound, and step. The loop is still statically countable
when those values depend on facts from the analysis context, such as a launch
coordinate. The analysis enumerates `scf.for` only when nested control flow
depends on its induction variable and the trip count fits the enumeration limit
defined above.

An operation with regions must implement `LoopLikeOpInterface` for loop
semantics, implement `RegionBranchOpInterface` for other structured control
flow, or provide a region-invocation evaluator. The analysis does not
infer execution semantics from traits such as `SingleBlock` or
`SingleBlockImplicitTerminator`; those traits constrain IR structure, not how
often a region executes.

The analysis runs MLIR sparse constant propagation and dead-code analysis once
for the root operation. These standard analyses provide context-independent
constant and executability facts. The per-query integer evaluator can prove
additional branch operands from consumer facts and enclosing induction values.
It also reads integer constants that sparse constant propagation forwards
through block and region arguments. Dead-code analysis determines whether a
block or edge may execute; it does not determine how many times it executes.

For each region and induction environment, the analysis constructs a possible
block control-flow graph (CFG). Its nodes are the region's blocks. Its edges
exclude branches proven dead by dead-code analysis and successors excluded by
compile-time branch selection. The graph preserves correlations between branch
arms and their merges.

A strongly connected component (SCC) is a maximal set of blocks in which every
block can reach every other block. An SCC is cyclic when it contains multiple
blocks or a single block with an edge to itself. A cyclic SCC is irreducible
when control flow can enter it through more than one block. A block
post-dominates the entry when every terminating or nonterminating continuation
from the entry passes through that block.

Within one region invocation, an unreachable block has count zero. A block in a
reachable cyclic SCC may execute more than once. The SCC may also never exit,
so every block reachable from it has an unknown count. Every other reachable
block executes at most once. Its count is one exactly when it post-dominates the
entry block in the possible CFG; otherwise, its count is unknown.

LLVM's SCC analysis identifies cycles. LLVM's post-dominator construction
treats ordinary exits and nonterminating loops as successors of a common
virtual exit. This distinction prevents a non-exiting cycle from incorrectly
forcing execution of a sibling block. A block before the branch can still
post-dominate the entry and have count one.

For example, each arm of a runtime-selected diamond has an unknown count, but
the common merge block has count one. A runtime branch that conditionally
enters one block leaves that block unknown while its unconditional merge still
has count one. A block after a reachable cycle remains unknown because the
cycle may execute repeatedly or never exit.

The analysis computes one SCC decomposition and one post-dominator tree for
each distinct possible block CFG in a region. It caches all block results from
that graph. Induction environments that select the same edges reuse the result.
All operations in one block execute once per block invocation, so queries for
operations in the same block also reuse the complete structured-control proof.

Structured loops remain supported through `LoopLikeOpInterface` because their
trip counts are modeled explicitly outside the block CFG analysis.

## Algorithm

In the pseudocode, `enumeration_budget` is the remaining number of loop
iterations that the analysis may inspect across all proof attempts. It is
initialized to the configured enumeration limit and is passed by reference, so
recursive calls consume the same budget. `try_consume` returns false when the
budget is zero; otherwise, it decrements the budget and returns true.
`empty_induction_environment` is an induction environment with no assigned loop
values. `frame_index` is a zero-based index into `control_frames`.
`enumerate_induction_values` computes the current loop's induction-variable
sequence from its lower bound, trip count, and step. Checked arithmetic returns
unknown on overflow.

For a loop frame, the recursive call first attempts the induction-independent
product. It uses the shared enumeration budget. Nested iterations remain
charged whether that attempt proves a count or falls back to enumerating the
current loop.

```text
exact_block_invocation_count(region, target_block, analysis_context,
                             induction_environment):
    graph = possible_block_cfg(region, analysis_context,
                               induction_environment)
    if graph is unknown:
        return unknown
    reachable = blocks_reachable_from(graph.entry)
    if target_block is not in reachable:
        return 0
    cyclic = blocks in reachable cyclic SCCs
    cycle_affected = blocks_reachable_from(cyclic)
    if target_block is in cycle_affected:
        return unknown
    if target_block post-dominates graph.entry:
        return 1
    return unknown

execution_count(operation, root_region, analysis_context):
    if operation is outside root_region:
        return unknown
    root_block, control_frames =
        enclosing_control_frames(root_region, operation)
    root_block_count = exact_block_invocation_count(
        root_region, root_block, analysis_context,
        empty_induction_environment)
    if root_block_count is unknown:
        return unknown
    if root_block_count == 0:
        return 0
    nested = count_frames(control_frames, 0, analysis_context,
                          empty_induction_environment, enumeration_limit)
    return checked_multiply(root_block_count, nested)

count_inside_frame(control_frames, frame_index, analysis_context,
                   induction_environment, enumeration_budget):
    frame = control_frames[frame_index]
    block_count = exact_block_invocation_count(
        frame.region, frame.block, analysis_context, induction_environment)
    if block_count is unknown:
        return unknown
    if block_count == 0:
        return 0
    nested = count_frames(control_frames, frame_index + 1, analysis_context,
                          induction_environment, enumeration_budget)
    return checked_multiply(block_count, nested)

count_frames(control_frames, frame_index, analysis_context,
             induction_environment, enumeration_budget):
    if frame_index == size(control_frames):
        return 1

    frame = control_frames[frame_index]

    if frame.region is not a loop body of frame.parent:
        invocations = exact_region_invocation_count(frame, analysis_context,
                                                     induction_environment)
        if invocations is unknown:
            return unknown
        if invocations == 0:
            return 0
        nested = count_inside_frame(control_frames, frame_index,
                                    analysis_context,
                                    induction_environment, enumeration_budget)
        if nested is unknown:
            return unknown
        return checked_multiply(invocations, nested)

    trip_count = exact_loop_trip_count(frame, analysis_context,
                                       induction_environment)
    if trip_count is unknown:
        return unknown
    if frame.parent has more than one loop region:
        return unknown
    if trip_count == 0:
        return 0

    nested = count_inside_frame(control_frames, frame_index, analysis_context,
                                induction_environment, enumeration_budget)
    if nested is exact:
        return checked_multiply(trip_count, nested)

    if frame.parent is not an scf.for:
        return unknown
    induction_values = enumerate_induction_values(
        frame, analysis_context, induction_environment)
    if induction_values is unknown:
        return unknown
    if size(induction_values) > enumeration_budget:
        return unknown

    total = 0
    for each induction_value in induction_values:
        if not try_consume(enumeration_budget):
            return unknown
        iteration_environment =
            induction_environment with frame's induction variable assigned
                                  to induction_value
        nested = count_inside_frame(control_frames, frame_index,
                                    analysis_context,
                                    iteration_environment, enumeration_budget)
        if nested is unknown:
            return unknown
        next_total = checked_add(total, nested)
        if next_total is unknown:
            return unknown
        total = next_total
    return total
```

## Toy operation-statistics client

`ttlang-op-stats` is a test-only toy client. It demonstrates how a consumer
can query the analysis and provides integration coverage for the public API; it
is not installed as a supported tool. The client groups operations by name
within each `func.func` and reports:

- `static_occurrences`: the number of operations with that name in the IR.
- `dynamic_instances`: the sum of their exact execution counts.

`dynamic_instances` is unknown if any contributing operation count is unknown
or the sum exceeds 64 bits. The tool sorts operation names so its output is
deterministic. It supplies no consumer-specific callback facts; counts that
require launch coordinates or custom region semantics therefore remain
unknown.

## Cost-model feature extraction

An optimization candidate is one legal IR configuration being compared with
other configurations. For each candidate, the analysis can construct a static
feature vector that maps every operation type to its total dynamic-instance
count:

```text
operation_instances[candidate][operation_type] =
    sum(execution_count(operation)
        for each operation of operation_type in candidate)
```

An entry in this vector is unknown if any contributing operation count is
unknown.

This supports cost models that parameterize a candidate by the counts of
lowered tile operations, initialization and reconfiguration operations, data
movement, and dataflow buffer (DFB) protocol operations. Let
`operation_cost[architecture][operation_type]` be the estimated cost of one
dynamic instance for a target architecture. A deterministic model can then
score an optimization candidate:

```text
static_work(candidate, architecture) =
    sum(operation_instances[candidate][operation_type]
        * operation_cost[architecture][operation_type]
        for each operation_type in candidate)
```

The compiler derives the candidate-specific instance counts from IR. It does
not need simulator traces or hardware measurements for each candidate.
Experimental data remains optional for calibrating or validating
`operation_cost`; the compiler can instead derive it from architecture
descriptions and documented instruction costs.

`static_work` is a work estimate, not elapsed time. Estimating elapsed time also
requires target-specific resource and scheduling facts, including:

- Which operations can execute concurrently.
- Resource occupancy and serialization constraints.
- Pipeline initiation intervals, meaning the number of cycles between the
  starts of successive iterations.
- Communication latency and contention.
- Dependencies that restrict overlap.

Keeping execution counts independent from those facts allows correctness
transformations to use exact counts without depending on a target cost model.
Optimization passes can use the same operation-instance features when comparing
legal candidates, then add latency, resource, and concurrency terms appropriate
to the decision.
