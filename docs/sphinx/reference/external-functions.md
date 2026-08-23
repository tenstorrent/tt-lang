# External Functions

`ttl.call_extern_func` invokes a C++ function declared in a custom header. It
supports static template arguments, runtime function arguments, an optional
typed scalar result, custom include directories, and portable logical-kernel
selection.

```python
ttl.call_extern_func(
    header,
    callee,
    *,
    template_args=None,
    func_args=None,
    dfb_dependencies=None,
    dfb_effects=None,
    unknown_dfb_access=False,
    include_paths=None,
    kernel=None,
    result_type=None,
    condition_result=None,
)
```

`header` and `callee` are compile-time strings. `template_args`, `func_args`,
and `include_paths` preserve source order. The compiler does not validate the
C++ signature.

## Scalar results

`result_type=ttl.ScalarType.I32` and `result_type=ttl.ScalarType.I64` declare
one signless scalar integer result. Omitting `result_type` or passing `None`
declares a void function. Raw strings, integers, and the `ScalarType` class are
invalid result declarations.

```python
predicate = ttl.call_extern_func(
    HEADER,
    "is_enabled",
    result_type=ttl.ScalarType.I64,
    kernel=ttl.KernelKind.COMPUTE,
)
if predicate:
    ttl.call_extern_func(
        HEADER,
        "execute_enabled_work",
        kernel=ttl.KernelKind.COMPUTE,
    )
```

A scalar-result call may be assigned directly in a unified operation. A call
selected for multiple logical kernels creates an independent local result in
each selected kernel. Enclosing structured control is retained only in logical
kernels that retain work in its regions. Composition preserves captured
`ScalarType` members and includes them in deterministic operation identity.

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

## Dispatch-stable condition results

`condition_result=` accepts a `ttl.DispatchCondition` created with
`ttl.ScalarType.I32` or `ttl.ScalarType.I64`. It identifies independent
evaluations of one runtime condition. Create the immutable declaration in an
enclosing operation factory and capture the same object in every evaluation:

```python
def make_conditional_operation():
    active = ttl.DispatchCondition(ttl.ScalarType.I64)
    producer = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    consumer = ttl.Kernel(ttl.KernelKind.COMPUTE)

    @ttl.operation(grid=(1, 1))
    def conditional_operation(input_tensor):
        producer_active = ttl.call_extern_func(
            HEADER,
            "evaluate_for_producer",
            condition_result=active,
            kernel=producer,
        )
        consumer_active = ttl.call_extern_func(
            HEADER,
            "evaluate_for_consumer",
            condition_result=active,
            kernel=consumer,
        )

    return conditional_operation
```

Calls using one declaration must return the same truth value for one dispatch
and launch coordinate. Zero is false and nonzero is true. Each evaluation must
be repeat-safe. A condition-result call cannot depend on DFB storage, declare
DFB effects, or set `unknown_dfb_access=True`. It cannot carry a DFB argument,
index, or descriptor. `condition_result` supplies the result type and cannot be
combined with `result_type`.

Composition and logical-kernel splitting preserve declaration identity.
Distinct declarations remain independent even when calls have equal C++ names,
headers, template arguments, or source text. The compiler also preserves
branch polarity, structured nesting, and supported boolean expressions when it
uses the identity to prove equal conditional DFB execution. Missing or partial
identity remains conservative. In the compiled IR module, equal condition
attributes identify one declaration and distinct declarations use distinct
ordinals.

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

When an external function consumes `ttl.get_dfb_id(dfb)`, the same DFB must be
a dependency through `func_args`, `ttl.dfb_descriptor(dfb)`, or
`dfb_dependencies`. An index value does not declare storage access by itself.

## DFB dependencies and protocol effects

`dfb_dependencies` declares DFB storage used by external C++ without adding
C++ function arguments. DFBs in `func_args` and DFB descriptors in
`template_args` are dependencies automatically. `dfb_dependencies` must
contain distinct DFBs that are not already automatic dependencies.

`dfb_effects` is an optional call-wide list of synchronous DFB protocol actions
in the exact order the external function executes them. Each action explicitly
names one DFB dependency and has a positive, statically resolvable tile count no
greater than the DFB capacity. A complete summary can provide the lifecycle
proof needed for physical-index reuse:

```python
ttl.call_extern_func(
    HEADER,
    "external_stage",
    template_args=[ttl.get_dfb_id(source)],
    func_args=[source],
    dfb_dependencies=[destination],
    dfb_effects=[
        ttl.DFBEffect.wait(source, tiles=2),
        ttl.DFBEffect.pop(source, tiles=2),
        ttl.DFBEffect.reserve(destination, tiles=1),
        ttl.DFBEffect.push(destination, tiles=1),
    ],
    kernel=ttl.KernelKind.DATA_MOVEMENT,
)
```

The example defines one sequence across both DFBs: wait on `source`, pop
`source`, reserve `destination`, then push `destination`. The per-DFB
subsequences are `wait -> pop` for `source` and `reserve -> push` for
`destination`. Actions on different DFBs occupy distinct positions in the
call-wide sequence.

The supported actions are `ttl.DFBEffect.reserve`, `push`, `wait`, and `pop`.
`ttl.DFBEffect.repeat(count, effects)` repeats a nonempty literal effect
sequence a nonnegative, statically resolvable number of times. The frontend
expands the repeat before creating IR, so downstream analyses receive the same
flat effect sequence as an explicitly written list. The expanded `dfb_effects`
sequence is limited to 4096 actions per external call. This compiler-resource
limit bounds materialization and analysis work; it is not a hardware limit:

```python
dfb_effects=[
    ttl.DFBEffect.repeat(
        transaction_count,
        [
            ttl.DFBEffect.wait(source, tiles=tiles_per_transaction),
            ttl.DFBEffect.pop(source, tiles=tiles_per_transaction),
        ],
    ),
]
```

Tile and repeat counts may use integer literals, integer captures, and
module-level integer variables combined with unary `+` or `-` and the binary
operators `+`, `-`, `*`, `//`, and `%`. Booleans and runtime SSA values are not
static integer counts. Floor-division and modulo divisors must be nonzero.

Every listed action occurs on every execution of the call, and list order is
execution order. Conditional actions must use TTL control flow around both the
matching acquisition and a call with an unconditional summary, execute
unconditionally in external C++, or be omitted so the dependency remains
opaque. Repeated transactions retain every action and its position. A bounded
lifecycle requires ordered reserve/push and wait/pop transactions with matching
tile counts. A partial summary is valid but does not prove a bounded lifecycle
for that dependency. A dependency occurrence with no listed effect is an opaque
storage access beginning at call entry, including when operand adaptation aliases
multiple occurrences to the same SSA DFB. Every aliased occurrence requires its
own effects to avoid an opaque access without a reset.

A named opaque dependency may retain protocol and asynchronous interface work
after the call. A synchronized reset ordered after the call through the same
participating logical kernel terminates that access and canonicalizes protocol
state. The reset implementation must complete earlier interface work before
publishing arrival. Storage reuse is permitted only after reset completion. This
does not validate the external function's internal queue protocol.

`unknown_dfb_access=True` declares that external C++ may access user-managed
DFBs not present in the declared dependencies. This is distinct from malformed
metadata. For allocation, the call becomes an opaque occurrence on every
user-managed DFB in each scope where it may execute, including listed DFBs.
Listed dependencies and effects remain available to other verification.

Every listed effect action is complete when the external function returns.
Associated interface work may remain active while the declared protocol retains
ownership; it must complete before the terminal consumer release or a
synchronized reset. Effects describe external behavior; they do not emit
reserve, push, wait, or pop calls.
Dependency-only operands and all effect metadata leave the generated C++ call
signature unchanged.

The IR stores each effect as a generated enum and a typed attribute. Its
dependency index identifies an element of the value sequence returned by the
call's `getDFBDependencyOperands()` interface method. Operation adaptation may
map distinct occurrences to the same DFB without merging them. Separate
executable operations would misrepresent actions already performed in C++,
while integer or string dictionaries would permit untyped effect kinds.
Callee-name, header-name, and generated-C++ inspection do not provide a semantic
contract and are not used.

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

Every `dfb_effects` action must complete before the external call returns. Its
associated interface work must complete before the terminal consumer release or
a synchronized reset. The compiler does not infer reserve, wait, push, or pop
operations from the C++ body; the `dfb_effects` contract supplies those facts
when required. A named dependency without effects remains opaque and requires a
synchronized reset before storage reuse.

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
and C++ retain the ordinary DFB push or pop signature; `kernel` is not an
emitted operand or argument. An explicit thread may use the same signature, but
its thread decorator already determines ownership, so the release selector has
no additional effect.
