# External Function Interop Lowering

## Overview

`ttl.call_extern_func` invokes a C++ function declared in a custom header. The
compiler preserves argument order, emits the header before the call, and lowers
each argument according to its source category. It does not inspect the C++
body or validate the foreign function signature.

External code must complete all synchronous and asynchronous resource accesses
before returning. The allocator therefore treats the call as one opaque access
interval, but the call does not reveal reserve/wait/push/pop effects implemented
inside C++.

The Python interface currently supports void calls.

## Logical-kernel selection

A `call_extern_func` in a unified `@ttl.operation` accepts a `kernel=` selector.
`KernelKind.COMPUTE` and `KernelKind.DATA_MOVEMENT` select the compiler-owned
canonical kernel of that kind:

```python
ttl.call_extern_func(
    HEADER,
    "compute_entry",
    kernel=ttl.KernelKind.COMPUTE,
)
```

Canonical kernel kinds may be combined with `|`:

```python
ttl.call_extern_func(
    HEADER,
    "shared_entry",
    kernel=ttl.KernelKind.COMPUTE | ttl.KernelKind.DATA_MOVEMENT,
)
```

An operation-local `Kernel` distinguishes multiple logical kernels of the same
kind. Its declaration is a static top-level operation resource:

```python
reader = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)

ttl.call_extern_func(HEADER, "reader_entry", kernel=reader)
ttl.call_extern_func(
    HEADER,
    "shared_entry",
    kernel=(ttl.KernelKind.COMPUTE, reader),
)
```

An external call accepts one selector or multiple distinct selectors. The `|`
syntax combines `KernelKind` values. A nonempty tuple supports selections that
include operation-local `Kernel` handles. Multiple selectors emit the call once
in every selected logical kernel. A call may omit `kernel=` when its enclosing
callback already determines one logical kernel. Otherwise, omission is invalid
because opaque code cannot be assigned by inspecting its implementation.

`TensorBlock.push` and `TensorBlock.pop` also accept `kernel=`, but only one
selector. An explicit selector assigns an otherwise-unused DFB transaction:

```python
unused = input_dfb.wait()
unused.pop(kernel=ttl.KernelKind.DATA_MOVEMENT)
```

`reserve` and `wait` do not accept a selector. Their ownership comes from the
acquired block's uses and release. The selector is consumed during unified-body
splitting and does not alter the external-call IR or C++ interface.

### Planning and identity invariants

Unified-body splitting analyzes immutable source AST before cloning or pruning
statements. Its immutable split plan records every statement selection, inferred
and explicit DFB transaction ownership, required kernel counts, and target
capacities. Split application consumes this plan without recomputing placement
from mutated AST. Retaining the required counts and capacities makes the
target-feasibility decision auditable after analysis.

Operation registration binds each `Kernel` handle in place exactly once to its
source name and owning operation. Equality and hashing require this binding and
include the kernel kind and complete logical identity. Equality or hashing of an
unbound handle is an error because Python object identity is not a stable logical
kernel identity. A deterministic fingerprint of immutable nonlocal captures
distinguishes factory-created operations whose generated code differs.
Composition retains the callee-owned bound handle, and repeated sequential calls
to the same callee share it. The bound identity is retained as typed function
metadata and remains available on `KernelSpec` after composition and core
specialization.

One target-indexed backend slot table supplies logical capacity validation,
logical-to-processor assignment, and final TTNN interop validation. Explicit and
unified operations therefore report capacity failures with the same logical
kernel kinds and identities.

## Operation runtime resources

### Motivation

TT-Metal separates reusable program structure from values supplied for each
dispatch. Kernel configuration, core placement, JIT definitions, program
semaphore layout, and the runtime-argument schema determine the program
structure. Per-core runtime-argument words may change when the cached program
executes again. Program execution may also depend on host or device objects
whose owners must remain alive through dispatch.

External TT-Lang kernels can require caller-defined program semaphores and JIT
definitions, per-core dispatch words, and host owners. These resources may
depend on the current tensors, device, launch range, and semaphore IDs already
reserved by the compiler. `runtime_resource_factory` supplies them for each
device execution without exposing target-specific kernel descriptor identities.

### Resource model

`@ttl.operation(runtime_resource_factory=...)` calls the factory once for each
device execution, before the runner constructs program descriptors. The
callback receives the current tensors, complete operation core range, and first
semaphore ID after the compiler-owned range. It returns frozen
`ProgramRuntimeResources` records with the following cache and lifetime
semantics:

| Resource | Role | Invocation and cache treatment |
| --- | --- | --- |
| `semaphore_descriptors` | Program semaphore structure. | IDs, ranges, core types, and initial values participate in cache identity. |
| `KernelDefine` | JIT compilation input. | Names and values participate in cache identity and apply to every specialized descriptor for the logical kernel. |
| `CoreRuntimeArgs` | Per-core dispatch values. | Logical destination, core, and vector length participate in cache identity; argument words may change on a cache hit. |
| `lifetimes` | Host ownership only. | Object identities do not participate in cache identity; references remain alive through execution. |

The factory runs for each device execution even when its structural result
selects an existing cached program. This preserves TT-Metal's distinction
between stable program structure and current dispatch values.

The [operation runtime resources reference](../sphinx/reference/operation-runtime-resources.md)
defines the callback signature, public records, and usage example.

### Planning and materialization

The runner validates the complete factory result and constructs a frozen
`ProgramResourcePlan` before creating any `KernelDescriptor` or
`ProgramDescriptor`. Planning validates record types, logical ownership, core
membership, unique semaphore IDs, the compiler/caller semaphore boundary, and
the unique destination of every per-core runtime argument. The plan contains
one resource entry for every compiled kernel descriptor, including descriptors
with no caller resources.

Descriptor construction consumes this plan without repeating selector lookup,
specialization routing, or cache policy. Planning does not modify compiled
kernel state or TTNN program state. A factory or planning failure occurs before
the runner constructs TTNN kernel or program descriptors and cannot produce a
partially materialized program.

### Logical identity and specialization

TT-Metal attaches definitions and runtime arguments to a `KernelDescriptor`.
TT-Lang does not expose descriptor indices as source identities: processor
assignment and descriptor order are target decisions, and core specialization
can materialize one logical kernel as several descriptors over disjoint core
sets. Resources therefore select the compiler-owned canonical kernel with
`KernelKind` or an explicit operation-owned kernel with `Kernel`.

The selected logical identity is retained on every `KernelSpec`, independent
of generated symbols, physical processors, and descriptor order. The planner
maps each resource to descriptors with the same identity.

The planner groups descriptors by logical identity and verifies that
specialized descriptors for one identity cover disjoint core sets. A
definition applies to every descriptor in the group because the descriptors
compile the same logical source. A per-core runtime argument applies only to
the descriptor whose core set contains that coordinate. Missing and multiple
destinations are errors.

### Program cache identity

The runner combines the compiled operation hash with a deterministic structural
fingerprint of the resource plan. Logical destinations, descriptor core sets,
definitions, runtime-argument coordinates and vector lengths, and caller
semaphore properties affect the fingerprint. Runtime-argument words, tensor
addresses, and lifetime object identities do not.

Changing a JIT definition or semaphore layout therefore selects a different
cached program. Changing only runtime-argument words reuses the same program
structure and supplies the current values to TT-Metal for that dispatch.

### Lifetimes and failures

Objects in `ProgramRuntimeResources.lifetimes` remain referenced through
execution. The factory result and its lifetime tuple remain local while the
runner plans, materializes, and executes the program. After successful
execution, the compiled operation stores the new tuple until a later successful
execution replaces it. A factory, planning, materialization, or execution
failure preserves the tuple from the last successful execution.

### Emitted runners and simulation

Emitted Python runners serialize logical identities and use the same planner
and materializer as decorated operation execution. A resource-aware emitted
runner requires the factory on every call because live resource objects and
dispatch values are not serialized.

Operation runtime resources are a hardware execution interface. The simulator
does not model TTNN program descriptors or per-core kernel runtime arguments
and rejects `runtime_resource_factory` as an unsupported `ttl.operation`
argument.

## Argument contract

| Source argument | Generated C++ interface | Restrictions |
| --- | --- | --- |
| `ttl.dfb_descriptor(dfb)` in `template_args` | `ttlang::DFBDescriptor<...>` template type | Declares a direct DFB dependency. |
| `ttl.get_dfb_id(dfb)` in `template_args` | Physical DFB index `uint32_t` literal | When the callee accesses DFB storage, the same DFB must declare a dependency through `func_args` or a `ttl.dfb_descriptor` template argument. |
| DFB in `func_args` | Physical DFB index `uint32_t` parameter | Declares a direct DFB dependency. |
| Integer or boolean in `template_args` | Signed integer or boolean constant | Must be compile-time evaluable. |
| Float in `template_args` | Unsigned IEEE-754 f32 bit-pattern constant | Must be compile-time evaluable. |
| Scalar in `func_args` | Lowered scalar parameter | Follows the TT-Metal kernel scalar convention. |
| Base tensor in `func_args` | `TensorAccessor` parameter | Supports tiled bf16 and fp32 tensors only in NOC kernels. |
| `ttl.raw_addr(base_tensor)` in `func_args` | `uint32_t` runtime tensor buffer address | Supports NOC and compute kernels; slices and derived tensor values are rejected. |
| Captured `ttnn.GlobalSemaphore` | `uint32_t` address literal or parameter | The address is fixed for the compiled operation. |

A bare DFB in `template_args` is ambiguous and rejected. The explicit wrapper
selects allocation metadata or an integer index.

## Typed DFB descriptors

A descriptor supplies the finalized physical allocation properties required by
an external DFB protocol:

```python
ttl.call_extern_func(
    HEADER,
    "external_copy",
    template_args=[
        ttl.dfb_descriptor(source),
        ttl.dfb_descriptor(destination),
    ],
    kernel=ttl.KernelKind.DATA_MOVEMENT,
)
```

The call lowers to a C++ template invocation equivalent to:

```c++
external_copy<
    ttlang::DFBDescriptor<0, 1, 2, 2048>,
    ttlang::DFBDescriptor<1, 1, 2, 2048>>();
```

The four parameters are the physical index, pages per block, block count, and
page size in bytes. The compiler emits the `ttlang::DFBDescriptor` definition
after `<cstdint>` and before custom headers, so a header can use its fields in
constant expressions:

```c++
template <typename Source, typename Destination>
inline void external_copy() {
  cb_reserve_back(Destination::index, Destination::pages_per_block);
  cb_wait_front(Source::index, Source::pages_per_block);
  // Copy and release the same page counts before returning.
}
```

Descriptor operands are direct DFB dependencies. They remain visible to DFB
lifetime and conflict analysis even though they become C++ types rather than
runtime function parameters.

## Integer DFB compatibility

`ttl.get_dfb_id` remains available for headers that accept integer template
indices:

```python
ttl.call_extern_func(
    HEADER,
    "legacy_copy",
    template_args=[ttl.get_dfb_id(source)],
    func_args=[source],
    kernel=ttl.KernelKind.DATA_MOVEMENT,
)
```

The ordinary DFB operand is required because an integer index argument does not
declare that the external function accesses DFB storage.

## Tensor arguments

A base tensor in `func_args` becomes a `TensorAccessor` containing the runtime
buffer address and compile-time accessor configuration. The interface supports
tiled bf16 and fp32 tensors in DRAM and L1. Only NOC kernels receive the
required accessor compile-time arguments. Compute and Ethernet kernels must use
a supported scalar interface instead.

`ttl.raw_addr(tensor)` reads the runtime tensor buffer address directly from the
NOC or compute kernel common arguments. It does not construct a
`TensorAccessor` or consume accessor compile-time arguments. The operand must be
an argument of the enclosing kernel-thread function with TTL layout encoding. A
nested region argument, slice, view, or computed tensor has no defined
runtime-argument mapping and is rejected. A raw address provides no layout,
view offset, page size, alignment, or bounds metadata.

## Global semaphores

Only a captured, single-address `ttnn.GlobalSemaphore` is accepted. The
frontend uses `ttnn.get_global_semaphore_address` and propagates getter errors.
It does not infer semaphore types from Python module names or try unrelated
address getters.

The wrapper retains the capture, and each operation wrapper has a distinct
compiler and program-cache identity. Runtime-varying semaphore addresses need a
separate runtime-argument interface.

## Lowering

The Python AST emits ordered typed attributes for static values and a separate
operand segment for referenced DFBs. TTL to TTKernel conversion resolves DFB
indices and materializes descriptor metadata before the DFB type loses its
block geometry. TTKernel to EmitC resolves constants and descriptor types, and
C++ emission inserts the required prelude and header during its existing
operation scan.

See `examples/external_dfb_reuse.py` for two external calls surrounded by
visible TTL protocol operations. An acknowledgment proves that the result
lifetimes do not overlap, while typed descriptor operands keep each opaque call
inside the corresponding lifetime.
