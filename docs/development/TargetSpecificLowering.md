# Target-Specific Lowering

This document defines how target-dependent compiler behavior is represented in
TT-Lang. Target selection must remain separate from operation matching and IR
mutation. A target implementation describes capabilities and lowering
decisions; it does not register competing rewrite patterns.

## Invariants

- Target-independent semantics remain in TTL and TTKernel operations.
- The target architecture is resolved once for each transformation scope.
- Analysis completes on immutable IR before a target-dependent rewrite begins.
- Every target decision is represented by a typed, inspectable plan.
- One deterministic rewrite implementation consumes each operation plan.
- Architecture-specific implementations override only decisions that differ.
- Omitting the target architecture is valid only when all registered targets
  produce the same plan.
- An architecture-specific implementation requires backend evidence and target
  coverage. A Blackhole-only implementation is not acceptable unless the
  Wormhole backend cannot implement the operation.

## Responsibility Boundaries

Target-dependent behavior is divided into three layers.

### Compute Capability

`ComputeTargetEnvironment` determines whether a compute operation and its tile
types are legal for an architecture. Queries receive target-independent typed
facts such as operand tile types, reduction kind, reduction dimension, and
broadcast kind. They return success or a diagnostic reason without modifying
IR.

Generic primitive checks cover properties shared by many operations. Exact
operation queries cover coupled constraints that cannot be inferred from one
tile type. For example, short-height reduction legality depends on the input,
scaler, result, reduction kind, and reduction dimension together.

### Kernel Configuration

`KernelTargetEnvironment` determines architecture-dependent compute-kernel
configuration. It owns limits and choices such as destination register
capacity, math fidelity support, and accumulation requirements. It does not
select TTKernel operations or emitted C++ APIs.

### Operation Lowering

When architectures require different TTKernel operations or algorithms, the
TTL-to-TTKernel conversion must use a `TTKernelTargetLowering` interface. This
interface is added when the first proven lowering difference is implemented;
unused dispatch interfaces are not added in advance.

The interface accepts immutable operation facts and returns typed plans. Its
conceptual API is:

```cpp
struct ReductionLoweringFacts {
  ttcore::TileType inputType;
  ttcore::TileType scalerType;
  ttcore::TileType resultType;
  ReduceType reduceType;
  ttkernel::ReduceDim reduceDimension;
};

struct ReductionLoweringPlan {
  ReductionImplementation implementation;
  ttkernel::ReduceDim reduceDimension;
  bool requiresScaler;
};

class TTKernelTargetLowering {
public:
  virtual PlanResult<ReductionLoweringPlan>
  planReduction(const ReductionLoweringFacts &facts) const;
};
```

`PlanResult<T>` has distinct planned, unsupported, and invalid-IR outcomes.
Unsupported means that the target cannot lower a valid operation. Invalid-IR
means that the operation violates dialect invariants. These outcomes must not
be represented by a null value or an empty optional.

A shared implementation contains the default Wormhole and Blackhole behavior.
An architecture subclass overrides an operation query only when its returned
plan differs:

```text
TTKernelTargetLowering
  |
  +-- WormholeBlackholeTargetLowering
        |
        +-- WormholeTargetLowering   (overrides proven differences only)
        |
        +-- BlackholeTargetLowering  (overrides proven differences only)
```

This structure keeps shared behavior in one implementation and makes each
architecture difference explicit. An override selects a typed strategy; it
does not directly create operations.

If only the emitted C++ API name or signature differs while TTKernel semantics
remain identical, specialization belongs in TTKernel-to-EmitC lowering. TTL
and TTKernel IR must not encode an ABI spelling difference.

## Planning Algorithm

A target-dependent conversion uses the following algorithm.

1. Resolve `ttl.target_arch` and the selected device architecture. Diagnose a
   disagreement before analysis.
2. Extract immutable target-independent facts for every candidate operation in
   the complete transformation scope.
3. Query the selected target implementation and record every decision in a
   `TileLoweringPlan`. Plans contain the operation, typed strategy, operands,
   result types, dependencies, and any required helper operations.
4. Validate that every operation requiring conversion has exactly one plan and
   that all plan dependencies are satisfiable.
5. Apply plans in deterministic IR order. The rewriter creates operations from
   the recorded strategy and does not recompute target policy.
6. Verify complete conversion and fail the pass if any source operation
   remains.

All planning and diagnostics precede the first mutation. A rejected
optimization leaves valid IR unchanged. A malformed operation produces a pass
failure rather than a pattern match failure after partial mutation.

The conversion registers one pattern for each source operation. Target
selection must not be implemented with multiple patterns whose benefits,
registration order, or match failures decide which architecture behavior is
used.

## Architecture-Independent Modules

When neither `ttl.target_arch` nor a selected device provides an architecture,
the compiler evaluates the operation for every registered target. Capability
validation succeeds only if every target accepts the operation. Lowering
planning succeeds only if every target returns an equality-comparable,
identical plan.

Different valid plans are not resolved by selecting an arbitrary common
strategy. The compiler emits a diagnostic requiring an explicit target. This
prevents architecture-independent compilation from silently changing generated
code when a new target is registered.

## Adding a Target Difference

An architecture override requires all of the following evidence:

1. A pinned backend revision showing the public API, instruction constraint,
   or hardware behavior that differs.
2. A shared implementation attempt or analysis explaining why the common
   strategy is invalid.
3. Positive lowering tests for the affected target.
4. Negative lowering tests for unsupported combinations.
5. Device tests on the affected target and regression tests on every target
   that continues to use the shared implementation.

For a proposed Blackhole-only implementation, Wormhole must be shown to lack a
required public API or to fail a focused device reproducer. Performance alone
does not justify different semantics; it may justify a target-specific typed
strategy when both implementations preserve the same operation contract.

## Subtile Reduction and Broadcast

Wormhole and Blackhole expose the same public LLK contracts for reduction
packing, reduction math, and unary broadcast at the pinned backend revision.
Both targets therefore share these capabilities:

- BF16 and F32 row sum and row max reduction with matching 8x32 input, scaler,
  and result tiles.
- BF16 and F32 column broadcast with matching 8x32 input and result tiles.

`ComputeTargetEnvironment` validates these coupled operation facts explicitly.
The existing target-independent TTL-to-TTKernel lowering then preserves the
8x32 tile type through reduction and broadcast. No architecture override is
required.
