# Compute Kernel Configuration

## Overview

Compute kernels share hardware configuration across all tile operations in a
`func.func`. DST element format and synchronization are kernel-wide. Each
dataflow buffer also has one unpack mode for the complete kernel. Tile
operations may support more than one execution strategy, and each strategy can
impose different requirements on those shared settings.

`ttl-set-compute-kernel-config` resolves these decisions before DST assignment:

```text
TTCore target attributes ---> KernelTargetEnvironment --+
pass options and attrs -----> KernelConfigPolicy --------+--> resolver
TTL tile operations --------> KernelRequirements --------+       |
                                                               v
                                                       KernelConfigPlan
                                                               |
                                                               v
                                               explicit IR attributes
```

The resolver separates target capabilities, compilation policy, and kernel
requirements. No component infers facts owned by another component.

## Tile Execution Semantics

Tile operations implement `TileExecutionOpInterface`. The interface reports:

- the hardware primitive;
- legal execution strategies;
- the route for each operand: dataflow buffer, DST, or no tile-data route;
- whether the result is resident in DST;
- the full-fp32 accumulation category, when configurable;
- whether repeated operations accumulate into an existing DST slot.

These facts are independent of the target. Target support is queried through
`KernelTargetEnvironment`.

Add, subtract, and multiply can use FPU or SFPU execution. FPU requires both
operands to address the same tile coordinates and resolve to dataflow buffers.
The selected strategy is written as `ttl.tile_execution_strategy` before DST
assignment. Later passes consume that attribute and do not recompute the
decision after copy insertion changes SSA operands.

An operation with no legal strategy alternatives provides fixed execution
semantics. A new tile operation participates by implementing the interface; the
kernel analysis does not maintain an operation-name list.

## Inputs

### Target Environment

`KernelTargetEnvironment` reads the typed `ttl.target_arch` attribute and the
existing TTCore system and device descriptions. If both sources are present,
their architectures must agree. A device selecting chips with different
architectures is invalid.

Capability queries contain architecture and backend restrictions. They consume
only target-independent execution categories, not operations. For example,
Blackhole row reduction and Wormhole reduction restrict full-fp32 accumulation,
and bf16 broadcast and transpose restrict f32 DST mode.

### Policy

`KernelConfigPolicy` normalizes pass options and function attributes. The
kernel-wide DST and synchronization settings use three-state selections:
`auto`, `enabled`, and `disabled`. Function attributes are hard constraints.
Full-fp32 reduce and matmul settings are preferences. The FPU binary setting
controls whether the FPU strategy is available.

An explicit `ttl.unpack_to_dest_fp32` attribute specifies the exact set of
dataflow buffer indices using that mode. It is validated against every
consumer.

### Requirements

The pass walks all nested regions without modifying IR. Fixed tile operations
contribute their requirements immediately. Operations with strategy
alternatives retain every legal option and its complete DFB and DST
requirements for joint resolution. The resolver does not re-query operations.

Each requirement records its operation and operand number for diagnostics. The
dataflow buffer resolver handles both `ttl.compute` block arguments and direct
tile operands derived from `tensor.extract` and `ttl.attach_cb`. A required
dataflow-buffer operand without a finalized DFB index is invalid.

## Resolution

The resolver maintains finite configuration domains:

```text
DST mode:             {default, fp32}
DFB N unpack mode:    {default, unpack-to-DST-fp32}
```

Hard policy constraints restrict the initial domains. Fixed operation
requirements are then intersected with the target-supported values. Each
strategy option is represented by the same requirement types as a fixed
operation.

Strategy selection and configuration selection are solved together. The
search chooses the unresolved operation with the fewest compatible options,
applies one option to a copy of the current domains, and continues until every
operation has a strategy. The resolver tries FPU before SFPU after hard
constraints are satisfied, preserving the default FPU preference without
depending on interface result order.

Joint resolution is required for correctness. A fixed SFPU consumer can require
unpack-to-DST-f32 for a DFB that is also read by an eligible binary operation.
Selecting FPU for the binary operation would require default unpack mode and
create a conflict. Selecting SFPU for the binary operation is valid. Resolving
the strategy before the shared DFB constraint would reject this valid kernel.

After all hard constraints are satisfied, reduce and matmul preferences select
f32 DST mode when it remains supported. Preferences never make a valid domain
empty.

DST synchronization currently has no operation-specific compatibility
constraints. `enabled` selects full synchronization; `auto` and `disabled`
select double-buffered synchronization.

An empty domain produces a diagnostic identifying both incompatible
requirements. The resolver retains typed conflict evidence rather than
reconstructing a cause from mutated IR.

## Application

Resolution returns a `KernelConfigPlan` containing:

- one selected strategy for each operation with alternatives;
- one DST mode;
- one DST synchronization mode;
- the sorted set of DFB indices using unpack-to-DST-f32.

Plan application validates all recorded operations before mutation. It then
writes operation strategy attributes and function configuration attributes
without additional failure points. The input-only
`ttl.enable_fpu_binary_ops` policy attribute is removed. Application derives no
additional policy.

## Correctness Invariants

- Target capabilities do not depend on kernel operations.
- Kernel requirements contain no architecture checks.
- Policy contains no inferred operation facts.
- Every DFB-consuming operand has an explicit route.
- One concrete unpack mode applies to each DFB for the complete kernel.
- One concrete DST mode and synchronization mode apply to the complete kernel.
- Strategy selection is stable across later IR rewrites.
- Failure before plan application leaves IR unchanged.
- Unknown execution semantics and unresolved DFB identities are errors.

## Pipeline Placement

The relevant ordering is:

```text
ttl-finalize-dfb-indices
ttl-set-compute-kernel-config
ttl-assign-dst
...
convert-ttl-to-ttkernel
```

DFB finalization must run first because the plan writes physical DFB indices to
`ttl.unpack_to_dest_fp32`. DST assignment must run after strategy selection
because operand routes determine copy insertion and register allocation.
TTKernel conversion consumes the same selected strategies.

Passes between strategy selection and conversion must preserve the selected
operand routes. A pass that cannot preserve them must reject the operation
before mutation.

## Extension

New tile operations add execution semantics through
`TileExecutionOpInterface`. Operations that introduce a new target restriction
also add a typed `KernelTargetEnvironment` query. This keeps operation
requirements independent from hardware capabilities and avoids adding another
whole-function classification scan.
