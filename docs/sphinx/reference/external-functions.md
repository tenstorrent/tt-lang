# External Functions

`ttl.call_extern_func` invokes a void C++ function declared in a custom header.
It supports static template arguments, runtime function arguments, custom
include directories, and portable logical-kernel selection.

```python
ttl.call_extern_func(
    header,
    callee,
    *,
    template_args=None,
    func_args=None,
    include_paths=None,
    kernel=None,
)
```

`header` and `callee` are compile-time strings. `template_args`, `func_args`,
and `include_paths` preserve source order. External functions currently return
no value, and the compiler does not validate the C++ signature.

## Logical-kernel selection

A unified `@ttl.operation` assigns an external call to one or more logical
kernels with `kernel=`. `KernelKind.COMPUTE` and
`KernelKind.DATA_MOVEMENT` select the compiler-owned canonical kernel of that
kind.

```python
@ttl.operation(grid=(1, 1))
def compute_external(inp):
    ttl.call_extern_func(
        HEADER,
        "compute_entry",
        func_args=[ttl.raw_addr(inp)],
        kernel=ttl.KernelKind.COMPUTE,
    )
```

Combine canonical kernel kinds with `|` when one call executes in both:

```python
ttl.call_extern_func(
    HEADER,
    "shared_entry",
    kernel=ttl.KernelKind.COMPUTE | ttl.KernelKind.DATA_MOVEMENT,
)
```

An operation-local `Kernel` distinguishes multiple kernels with the same
kind. An operation factory may create one handle and capture it in the
operation and related factory callbacks. Operation registration binds the
same handle in place using its capture name and deterministic operation
identity.

```python
def make_selected_external():
    reader = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)

    @ttl.operation(grid=(1, 1))
    def selected_external(inp):
        ttl.call_extern_func(
            HEADER,
            "reader_entry",
            func_args=[inp],
            kernel=reader,
        )
        ttl.call_extern_func(
            HEADER,
            "shared_entry",
            kernel=(ttl.KernelKind.COMPUTE, reader),
        )

    return selected_external
```

A handle used only by the operation may instead be declared as a static
top-level assignment in the operation body.

Explicit multi-kernel operations use the same selectors on thread decorators.
Each decorator accepts one selector whose kind matches the thread type. An
omitted selector denotes the canonical kernel of that kind.

```python
def make_explicit_operation():
    reader = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)

    @ttl.operation(grid=(1, 1))
    def explicit_operation(inp):
        @ttl.compute(kernel=ttl.KernelKind.COMPUTE)
        def compute_thread():
            pass

        @ttl.datamovement(kernel=reader)
        def reader_thread():
            pass

    return explicit_operation
```

Operation registration binds a captured thread selector before compilation.
This permits the same handle to identify the compiled kernel in runtime
configuration APIs.

Composing a unified operation preserves each callee-owned handle, including
its operation identity. Repeated sequential calls to the same composed
operation share that logical kernel instead of consuming additional target
kernel resources. The original handle therefore remains equal to the
`KernelSpec.logical_kernel` value produced for the composed program.
Factory-created operations with different immutable nonlocal captures receive
different deterministic operation identities. Equal captures retain the same
identity.

An external call accepts one selector or multiple distinct selectors. The `|`
syntax combines `KernelKind` values. A nonempty tuple supports selections that
include operation-local `Kernel` handles. Multiple selectors emit the call once
in every selected logical kernel. A call may omit `kernel=` when its enclosing
callback already determines one logical kernel. A top-level opaque call without
a selector is invalid because the compiler cannot infer placement from C++
code.

The target backend assigns logical kernels to its supported kernel resources.
Compilation fails when an operation requests more kernels of a kind than the
target supports. Unified and explicit multi-kernel operations use the same
target capacity table and diagnostic terms.

## Template arguments

`template_args` accepts compile-time values and explicit DFB wrappers.

| Python argument | Generated C++ argument |
| --- | --- |
| `int` | Signed integer constant |
| `bool` | Boolean constant |
| `float` | Unsigned binary32 bit-pattern constant |
| `ttl.dfb_descriptor(dfb)` | `ttlang::DFBDescriptor<index, pages_per_block, block_count, page_size>` type |
| `ttl.get_dfb_id(dfb)` | Physical DFB index constant |

A bare DFB is invalid in `template_args`. `ttl.dfb_descriptor` supplies typed
allocation metadata. `ttl.get_dfb_id` supplies only an integer index.

```python
ttl.call_extern_func(
    HEADER,
    "external_copy",
    template_args=[
        ttl.dfb_descriptor(source_dfb),
        ttl.dfb_descriptor(destination_dfb),
        4,
        False,
    ],
    kernel=ttl.KernelKind.DATA_MOVEMENT,
)
```

## Function arguments

`func_args` accepts lowered scalar values, DFBs, base tensors, and raw tensor
addresses.

| Python argument | Generated C++ argument | Restrictions |
| --- | --- | --- |
| Scalar value | Scalar parameter | Uses the kernel runtime-argument convention. |
| DFB | Physical DFB index parameter | Declares a direct dependency on that DFB. |
| Base tensor | `TensorAccessor` parameter | Supported in data-movement kernels for tiled BF16 and FP32 tensors. |
| `ttl.raw_addr(tensor)` | `uint32_t` buffer address | Supported in compute and data-movement kernels. |

Tensor slices, views, and computed tensor values are not valid external
arguments. `ttl.raw_addr` provides no layout, view offset, page size, alignment,
or bounds metadata.

When `ttl.get_dfb_id(dfb)` identifies storage accessed by the external
function, the same DFB must appear as a direct dependency through `func_args`
or `ttl.dfb_descriptor(dfb)`.

## Include directories

`include_paths` contains compile-time directory strings added to external
header lookup. The compiler emits the requested header before the call.

```python
ttl.call_extern_func(
    "kernels/custom_entry.hpp",
    "custom_entry",
    include_paths=[PROJECT_INCLUDE_DIR],
    kernel=ttl.KernelKind.COMPUTE,
)
```

## DFB synchronization ownership

External C++ must complete its resource accesses before returning. The compiler
does not infer reserve, wait, push, or pop operations from the C++ body.

`TensorBlock.push` and `TensorBlock.pop` accept one `kernel=` selector when a
DFB transaction has no other use from which ownership can be inferred.

```python
unused = input_dfb.wait()
unused.pop(kernel=ttl.KernelKind.DATA_MOVEMENT)
```

`reserve` and `wait` do not accept `kernel=`. Their logical-kernel ownership is
derived from the acquired block's uses and release. An explicit release
selector that conflicts with inferred ownership is invalid.

The release selector affects unified-operation splitting only. Generated IR
and C++ retain the ordinary argument-free DFB release operation. An explicit
thread may use the same signature, but its thread decorator already determines
ownership, so the release selector has no additional effect.
