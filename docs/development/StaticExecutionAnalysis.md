# Static Execution Analysis

This document defines the compile-time analysis used to count operation
executions within structured control flow. PipeNet verification is the first
consumer. The same result can support cost models that combine operation counts
with latency and concurrency information.

## Semantics

The analysis answers this question:

> How many dynamic instances of an operation execute during one invocation of
> a specified root region?

The root region is part of the query. For a kernel function, its body is the
root region and the result is relative to one function invocation. A callee is
analyzed relative to one invocation of its own body. Interprocedural consumers
must compose callee counts with call-site counts.

The result is either:

- An exact unsigned 64-bit count, including zero.
- Unknown because an exact count is not proven.

Unknown is distinct from zero and one. A correctness transformation that
requires an exact count must reject an unknown result. A performance model may
preserve the unknown value or combine it with a separate symbolic or range
analysis.

An operation that executes in a compile-time-evaluable loop has an exact count
even though its instances execute at runtime. A count is unknown when it
depends on runtime data that the analysis context cannot evaluate, such as a
function argument, a memory load, or a loop-carried value without a proven
constant.

The count describes dynamic operation instances. It does not describe their
execution order, latency, parallelism, or overlap. For example, a parallel loop
with 64 iterations contributes 64 instances, but this analysis does not assert
whether those instances execute concurrently.

## Control-flow representation

For an operation nested under structured control flow, the analysis records the
enclosing regions from the root region to the operation. Each parent operation
contributes the number of times it invokes the selected child region.

For an induction-independent nest:

```text
operation_count = product(region_invocation_count(frame))
```

The product is valid only when every factor is exact. An exact zero ends the
proof because the nested operation is unreachable in that analysis context.

When a selected region depends on an enclosing induction variable, the count
is a sum over the statically enumerable iterations:

```text
operation_count = sum(count_of_nested_frames(iteration))
```

This distinction avoids enumerating large rectangular loop nests while still
handling compile-time-evaluable predicates such as
`if iteration < static_limit`.

## MLIR interfaces

The analysis uses MLIR interfaces as the primary semantic source:

- `LoopLikeOpInterface::getStaticTripCount()` supplies an exact loop trip
  count when the loop operation can compute one. The analysis requires one
  loop region because the interface does not define separate invocation counts
  for regions in a multi-region loop.
- `RegionBranchOpInterface::getRegionInvocationBounds()` supplies region
  invocation bounds. A region count is exact only when its lower and upper
  bounds are equal.

Constant operands are passed to `getRegionInvocationBounds()`. Integer
operands that become constant in the current analysis context are also passed
as attributes.

Some upstream structured branch operations retain a conservative zero lower
bound even when their selector is constant. The analysis combines invocation
bounds with `getEntrySuccessorRegions()` and region successors. One selected
entry region executes exactly once when its upper bound is one, or when it can
only return to the parent operation. This proves structured selection without
operation-specific condition handling. The analysis does not otherwise treat
a region's upper bound as exact; doing so for an arbitrary
`RegionBranchOpInterface` would be unsound.

When a constant selector identifies another entry region, the analysis walks
the interface successor graph. A target region has count zero only when no
successor sequence from the selected region can reach it.

`scf.for` receives additional support for context-evaluable lower bounds, upper
bounds, and steps. The loop is still statically countable when those values
depend on compile-time analysis symbols, such as a launch coordinate. The
analysis enumerates `scf.for` only when nested control flow depends on its
induction variable and the trip count fits the configured proof limit.

An operation with regions must implement the relevant interface or provide a
context-specific region evaluator. The analysis does not infer execution
semantics from traits such as `SingleBlock` or
`SingleBlockImplicitTerminator`; those traits constrain IR structure, not how
often a region executes.

## Analysis context

Some compile-time values are known only to a consumer. The analysis accepts two
callbacks:

- A symbol-value evaluator returns an integer value for context-specific SSA
  values. Its result must have the same bit width as the SSA value. PipeNet
  analysis uses it for launch coordinates and PipeNet role predicates.
- A region-invocation evaluator returns an exact count for context-specific
  non-loop region semantics. PipeNet analysis uses it for coordinate-dependent
  `ttl.if_src` and `ttl.if_dst` regions and for the unconditional
  `ttl.pipenet_scope` region.

The shared analysis evaluates constants, induction variables, integer casts,
addition, subtraction, multiplication, bitwise boolean operations, and integer
comparisons. Consumers do not duplicate those rules.

Callbacks return facts, not defaults. An unevaluable symbol or region remains
unknown and is processed through the standard MLIR interfaces when possible.

## Algorithm

```text
execution_count(operation, root_region, context):
    frames = enclosing_control_frames(root_region, operation)
    if operation is outside root_region:
        return unknown
    if a relevant region contains arbitrary block control flow:
        return unknown
    return count_frames(frames, 0, empty_induction_environment)

count_frames(frames, frame_index, induction_environment):
    if frame_index == frames.size:
        return 1

    frame = frames[frame_index]

    if frame is not a loop body:
        invocations = exact_region_invocation_count(frame, context,
                                                     induction_environment)
        if invocations is unknown:
            return unknown
        if invocations == 0:
            return 0
        nested = count_frames(frames, frame_index + 1,
                              induction_environment)
        return checked_multiply(invocations, nested)

    trip_count = exact_loop_trip_count(frame, context,
                                       induction_environment)
    if trip_count is unknown:
        return unknown
    if trip_count == 0:
        return 0

    nested = count_frames(frames, frame_index + 1,
                          induction_environment)
    if nested is exact:
        return checked_multiply(trip_count, nested)

    if frame is not an enumerable scf.for:
        return unknown
    if enumeration exceeds the proof limit:
        return unknown

    total = 0
    for each statically computed induction value:
        nested = count_frames(frames, frame_index + 1,
                              induction_environment + induction_value)
        total = checked_add(total, nested)
    return total
```

All addition and multiplication are checked. Overflow produces unknown rather
than a wrapped count.

## Arbitrary block control flow

Region invocation counts do not determine how often each block inside a region
executes. Until a block-frequency proof is available, the analysis requires one
block with no block successors in every relevant region. It returns unknown for
multi-block control flow. This prevents a conditional branch or loop backedge
from being treated as one execution.

A future extension can compute exact block counts from `BranchOpInterface`
successors when edge selection and backedge counts are compile-time evaluable.
That result composes with the region and loop factors defined above without
changing the public count semantics.

## Cost-model feature extraction

The analysis can construct a static feature vector by aggregating exact counts
for each operation type:

```text
operation_instances[operation_type] =
    sum(execution_count(operation) for each operation of operation_type)
```

This matches the compiler cost-model strategy of parameterizing a candidate by
operation class, lowered tile operations, initialization and reconfiguration
operations, data movement, and DFB protocol operations. A deterministic model
can score an optimization candidate with target-specific per-operation
parameters:

```text
static_work(candidate) =
    sum(operation_instances[operation_type]
        * operation_cost[architecture][operation_type])
```

The compiler derives the candidate-specific instance counts from IR. It does
not need simulator traces or hardware measurements for each candidate.
Experimental data remains optional for calibrating or validating the
per-operation parameters; an analytical model can instead use target
descriptors and documented instruction costs.

The sum is a work estimate, not elapsed time. Estimating elapsed time also
requires target-specific resource and scheduling facts, including:

- Which operations can execute concurrently.
- Resource occupancy and serialization constraints.
- Pipeline initiation intervals.
- Communication latency and contention.
- Dependencies that restrict overlap.

Keeping execution counts independent from those facts allows correctness
transformations to use exact cardinality without depending on a target cost
model. Cost models for subblock selection, fusion, and block planning can use
the same operation-instance features when comparing legal candidates, then add
latency, resource, and concurrency terms appropriate to the decision.

## Required coverage

Tests must cover:

- Straight-line operations with count one.
- Constant and context-evaluable loop bounds and predicates.
- Consumer symbol values with valid and invalid bit widths.
- Consumer-defined non-loop region invocation counts.
- Large induction-independent rectangular loop nests without enumeration.
- Induction-dependent branches with exact enumerated counts.
- Exact zero for an unreachable region or zero-trip enclosing loop.
- Signed and unsigned loop comparison semantics.
- Dynamic bounds or conditions producing unknown.
- Unsupported loop and region semantics producing unknown.
- Multi-block region control flow producing unknown.
- Enumeration-limit boundaries and arithmetic-overflow handling.

PipeNet regression tests additionally verify that send and receiver-post
counts match for compile-time-evaluable loop and branch combinations, and that
unknown or unequal counts produce a diagnostic instead of generated code.
