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
