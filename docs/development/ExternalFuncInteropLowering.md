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
