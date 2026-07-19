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

An operation in a loop has an exact count when the analysis proves the loop's
iteration count. A count is unknown when it depends on runtime data that the
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
iteration. The shared analysis evaluates constants, loop induction variables,
integer casts, addition, subtraction, multiplication, bitwise AND, OR, and XOR,
and integer comparisons. Consumers do not duplicate those rules.

Callbacks return facts, not defaults. If a callback supplies no fact, the
analysis uses the standard Multi-Level Intermediate Representation (MLIR)
interfaces when possible. The result remains unknown when neither mechanism
proves an exact value or count.

## Control-flow representation

A control frame pairs a parent operation with one of its child regions. Its
region invocation count is the number of times the parent invokes that child
region. For an operation nested under structured control flow, the analysis
records an ordered control-frame sequence from the root region to the
operation. This document names that sequence `control_frames`.

A control-frame sequence is induction-independent when no frame's invocation
count depends on an induction variable from an earlier loop frame. Its
operation count is the product of the frame invocation counts:

```text
operation_count =
    product(region_invocation_count(frame) for frame in control_frames)
```

The product is valid only when every factor is exact. An exact zero ends the
proof because the nested operation is unreachable in that analysis context.

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

## Algorithm

In the pseudocode, `enumeration_budget` is the remaining number of loop
iterations that the analysis may inspect. It is initialized to the configured
enumeration limit and is passed by reference, so recursive calls consume the
same budget. `empty_induction_environment` is an induction environment with no
assigned loop values. A relevant region is the root region or a child region in
`control_frames`. `frame_index` is a zero-based index into `control_frames`.
`enumerate_induction_values` computes the current loop's induction-variable
sequence from its lower bound, trip count, and step. Checked arithmetic returns
unknown on overflow.

For a loop frame, the recursive call first attempts the
induction-independent product. It receives a copy of the enumeration budget. A
successful proof commits any budget consumed by nested loops; a failed proof
discards that consumption before enumerating the current loop.

```text
execution_count(operation, root_region, analysis_context):
    if operation is outside root_region:
        return unknown
    control_frames = enclosing_control_frames(root_region, operation)
    if any relevant region has multiple blocks or block successors:
        return unknown
    return count_frames(control_frames, 0, analysis_context,
                        empty_induction_environment, enumeration_limit)

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
        nested = count_frames(control_frames, frame_index + 1,
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

    independent_budget = enumeration_budget
    nested = count_frames(control_frames, frame_index + 1, analysis_context,
                          induction_environment, independent_budget)
    if nested is exact:
        enumeration_budget = independent_budget
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
        enumeration_budget = enumeration_budget - 1
        iteration_environment =
            induction_environment with frame's induction variable assigned
                                  to induction_value
        nested = count_frames(control_frames, frame_index + 1,
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

## Multi-block control flow

A block successor is another block that a block terminator may execute next.
Region invocation counts do not determine how often each block inside a region
executes. Until the analysis can prove an exact execution count for every
block, it requires one block with no block successors in every relevant region.
It returns unknown for multi-block control flow. This prevents a conditional
branch or a branch to an earlier block from being treated as one execution.

Supporting multi-block regions requires exact block counts computed from
`BranchOpInterface` successors. The analysis must determine which successor
executes and how often each branch to an earlier block executes. Those block
counts compose with the region and loop factors defined above without changing
the definition of an operation count.

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

## Test coverage

Tests cover:

- Straight-line operations with count one.
- Loop bounds and predicates that are constant or evaluable from
  facts in the analysis context.
- Symbol-value evaluator results with matching and mismatched SSA bit widths.
- Consumer-defined non-loop region invocation counts.
- Large rectangular loop nests without enumeration.
- Induction-dependent branches with exact enumerated counts.
- Exact zero for an unreachable region or zero-trip enclosing loop.
- Signed and unsigned loop comparison semantics.
- Loop bounds or branch conditions that depend on unevaluated runtime values
  producing unknown.
- Loops and region operations without a supported exact-count model producing
  unknown.
- Multi-block region control flow producing unknown.
- Enumeration-limit boundaries and arithmetic overflow handling.

In a receiver-post PipeNet protocol, each receiver announces that it has
reserved destination storage. PipeNet regression tests additionally verify
that sender-transfer and receiver-post operation counts match for loops and
branches that the analysis can evaluate, and that unknown or unequal counts
produce a diagnostic instead of generated code.
