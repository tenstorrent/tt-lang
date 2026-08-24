# Compute Kernel Configuration

## Overview

Compute kernels share hardware configuration across all tile operations in a
`func.func`. Destination element width and synchronization are kernel-wide.
Each dataflow buffer also has one target-defined unpack setting for the complete
kernel. Tile operations may support more than one execution strategy, and each
strategy can impose different requirements on those shared settings.

`ttl-set-compute-kernel-config` resolves these decisions before DST assignment:

```text
TTCore target attributes ---> KernelTargetEnvironment ---+
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
- which DST operands the operation lowering initializes itself;
- whether the result is resident in DST;
- the maximum number of simultaneously resident DST slots;
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

When neither source is present, the compiler uses the current shared Wormhole
B0/Blackhole configuration relations without architecture-specific
restrictions. This environment is limited to compiler testing. Pipelines that
emit device kernels must attach an architecture before this pass.

Capability queries contain architecture and backend restrictions. They consume
only target-independent execution categories, element types, and operand
routes, not operation classes. The target returns allowed combinations of
destination element width and DFB unpack mode. It does not return a preferred
singleton. Target lookup constructs a target-specific implementation of the
query interface; architecture dispatch does not occur inside individual
queries. Adding an architecture therefore requires an explicit implementation
rather than inheriting another target's behavior.

The current TT-Lang runtime launches Wormhole B0 and Blackhole through
`ttnn.ComputeConfigDescriptor`. Both expose 16-bit and 32-bit destination
elements. Their current broadcast LLKs restrict non-32-bit broadcast inputs to
16-bit destination elements. Blackhole row reduction and Wormhole reduction
also restrict full-fp32 accumulation.

Quasar requires the Gen2 configuration descriptor, global unpack routing, and
Quasar kernel launch mechanism. The current TT-Lang runtime does not implement
those interfaces, so this pass rejects Quasar. Recognizing a TTCore architecture
enum is not sufficient to compile or launch a kernel for that target.

### Policy

`KernelConfigPolicy` normalizes pass options and function attributes. The
kernel-wide DST and synchronization settings use three-state selections:
`auto`, `enabled`, and `disabled`. Function attributes are hard constraints.
Full-fp32 reduce and matmul settings are preferences. The FPU binary setting
controls whether the FPU strategy is available.

The operation-level `math_fidelity` selection is forwarded to every generated
TTNN compute descriptor. It does not participate in joint resolution because
it does not change the current tile-strategy, DST, or DFB compatibility
relations.

An explicit `ttl.unpack_to_dest_fp32` attribute specifies the exact set of
dataflow buffer indices using `ComputeConfigDescriptor::unpack_to_dest_mode`.
It is validated by intersecting the selected setting with every consumer's
allowed target configurations. The current runtime accepts entries for
non-f32 DFBs as inert, so the analysis does not reject those entries.

### Requirements

The pass walks all nested regions without modifying IR. Fixed tile operations
contribute their requirements immediately. Operations with strategy
alternatives retain every legal option and its complete DFB and DST
requirements for joint resolution. The resolver does not re-query operations.

Each requirement records its operation and operand number for diagnostics. The
dataflow buffer resolver handles both `ttl.compute` block arguments and direct
tile operands derived from `tensor.extract` and `ttl.attach_cb`. Tile operands
and tensors of tiles use the same element-type query, including the tensor form
produced for matmul after loop lowering. A required dataflow-buffer operand
without a finalized DFB index is invalid.

Destination requirements use the result or resident operand type, not only DFB
input storage width. The width query uses the TTCore tile element type, so a
supported operation with 32-bit integer destination elements requires 32-bit
destination registers without an integer-type exception in this analysis.
Fixed-block operations also record their simultaneous DST residency. This
requirement remains explicit when tensor operands are scalarized to tile values.

## Resolution

The resolver maintains a finite set of complete target configurations:

```text
destination element width:     {Bits16, Bits32}
destination synchronization:   {DoubleBuffered, Full}
DFB N unpack mode:              {Default, UnpackToDestination}
```

Not every Cartesian-product combination is legal. Each target query returns
the allowed width-and-mode relation for a primitive, element type, and operand
route. `TileOperandRoute` separately records the primitive's physical operand
route; the unpack mode does not describe that route for formats where the
setting is inert. Hard policy constraints restrict the initial candidates,
including an explicit synchronization setting. Fixed operation requirements
are then intersected with the target-supported relations and the DST capacity
for each width-and-synchronization pair. Each strategy option is represented by
the same requirement types as a fixed operation.

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
32-bit destination elements when they remain supported. Preferences never make
a valid domain empty.

DST synchronization affects the number of available DST slots. `enabled` and
`disabled` select full and double-buffered synchronization, respectively.
`auto` prefers double-buffered synchronization when it satisfies every fixed
residency requirement and otherwise selects full synchronization.

An empty domain produces a diagnostic identifying incompatible requirements.
A capacity conflict reports the required and maximum available slot counts.
The resolver retains typed conflict evidence rather than reconstructing a cause
from mutated IR.

## Application

Resolution returns a `KernelConfigPlan` containing:

- one selected strategy for each operation with alternatives;
- one destination element width;
- one DST synchronization mode;
- the sorted set of DFB indices using unpack-to-DST-f32.

Only the resolver can construct a plan. Plan application writes operation
strategy attributes and function configuration attributes without re-deriving
policy or introducing additional failure points. The input-only
`ttl.enable_fpu_binary_ops` policy attribute is removed. Application derives no
additional policy.

## Correctness Invariants

- Target capabilities do not depend on kernel operation classes.
- Kernel requirements contain no architecture checks.
- Policy contains no inferred operation facts.
- Every DFB-consuming operand has an explicit route.
- One target-supported unpack setting applies to each DFB for the complete
  kernel.
- One concrete destination width and synchronization mode apply to the complete
  kernel.
- Every fixed-block residency fits the selected destination capacity.
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
extend the typed target queries with an allowed configuration relation. A new
architecture adds its own query implementation and runtime translation; it
does not inherit Gen1 behavior by default. Runtime names such as
`fp32_dest_acc_en` and `ttl.unpack_to_dest_fp32` remain at the IR/runtime
translation boundary. Operation-only settings without compiler compatibility
relations, such as `math_fidelity`, remain runtime descriptor inputs.

### Adding an Architecture or Configuration API

Architecture support requires all of the following changes:

1. Add a `KernelTargetEnvironment` implementation that defines every
   destination-width, per-DFB routing, and full-precision accumulation query
   for the architecture and configuration API used by the compiler pipeline.
   Reuse a base class only when the lower-level configuration and LLK contracts
   are identical for every inherited query.
2. Register that implementation in `KernelTargetEnvironment::get`. This is the
   only architecture switch in configuration analysis. An architecture without
   an implementation must produce a diagnostic during target construction.
3. Add the corresponding compiler-to-runtime configuration translation and
   kernel launch mechanism. Target recognition alone is not runtime support.
4. Map runtime device discovery to the typed TTCore architecture attribute.
   Unknown runtime architectures must fail instead of selecting another
   target's implementation.
5. Add compiler tests for every target-specific query difference, invalid
   configuration diagnostics, and architecture/device disagreement. Add device
   tests that verify the emitted configuration through the target's actual
   runtime descriptor and launch mechanism.

The implementation must be derived from the pinned tt-metal contract for that
architecture. New fields remain target-neutral inside requirements and the
plan when they represent shared hardware concepts. Generation-specific runtime
spellings remain in the translation layer.

Changing the configuration or launch API for an existing architecture also
requires reviewing every target query. Architecture identity alone is not a
compatibility proof. For example, the pinned experimental Metal 2.0
`ComputeGen1Config` validates some inert `unpack_to_dest_mode` entries more
strictly than the `ComputeConfigDescriptor` used by the current TT-Lang
runtime. If a Wormhole B0 or Blackhole pipeline changes to `ComputeGen1Config`,
target construction must select a `KernelTargetEnvironment` for that API. It
must not select the existing `ComputeConfigDescriptor` environment solely from
the unchanged architecture enum. Shared query implementations require a
query-by-query compatibility review of validation and lowering semantics.

Configuration analysis is independent of physical tile height and width. The
same requirements are collected for every tile dimension supported by a
primitive. This does not expand primitive support: the lowering and LLK must
still support the selected physical tile dimensions.
