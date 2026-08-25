# External Function Interop Lowering

## Overview

`ttl.call_extern_func` invokes a C++ function declared in a custom header. The
compiler preserves argument order, emits the header before the call, and lowers
each argument according to its source category. It does not inspect the C++
body or validate the foreign function signature.

DFB behavior is declared rather than inferred: `ttl.opaque_call` exposes
dependencies, ordered protocol effects, and unknown access through
`DFBAccessOpInterface`. Every declared effect action completes before return,
but associated asynchronous interface work may remain live until the terminal
consumer release or a synchronized reset. A dependency occurrence with no
listed effect remains opaque until a synchronized reset proves completion. A
complete effect summary can establish the lifecycle facts required for
physical-index reuse.

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
| `ttl.get_dfb_id(dfb)` in `template_args` | Physical DFB index `uint32_t` literal | When the callee accesses DFB storage, the same DFB must declare a dependency through `func_args`, `ttl.dfb_descriptor`, or `dfb_dependencies`. |
| DFB in `func_args` | Physical DFB index `uint32_t` parameter | Declares a direct DFB dependency. |
| DFB in `dfb_dependencies` | No generated C++ argument | Declares dependency-only storage access. Entries must be distinct and must not duplicate automatic dependencies. |
| Integer or boolean in `template_args` | Signed integer or boolean constant | Must be compile-time evaluable. |
| Float in `template_args` | Unsigned IEEE-754 f32 bit-pattern constant | Must be compile-time evaluable. |
| Scalar in `func_args` | Lowered scalar parameter | Follows the TT-Metal kernel scalar convention. |
| Base tensor in `func_args` | `TensorAccessor` parameter | Supports tiled bf16 and fp32 tensors only in NOC kernels. |
| `ttl.raw_addr(base_tensor)` in `func_args` | `uint32_t` runtime tensor buffer address | Supports NOC and compute kernels; slices and derived tensor values are rejected. |
| Captured `ttnn.GlobalSemaphore` | `uint32_t` address literal or parameter | The address is fixed for the compiled operation. |

A bare DFB in `template_args` is ambiguous and rejected. The explicit wrapper
selects allocation metadata or an integer index.

## DFB dependency and effect representation

The [external-functions reference](../sphinx/reference/external-functions.md)
defines the Python API and static-expression rules. Every statically known DFB
accessed by external code must be declared as a dependency. The `dfb_effects`
summary is optional: without it, each dependency remains an opaque access until
a synchronized reset proves completion. A complete, accurate summary can prove
a bounded lifecycle and permit physical-index reuse.

`OpaqueCallOp::getDFBDependencyOperands()` returns dependency occurrences in
this order:

1. DFB occurrences in `func_args`.
2. DFB descriptor occurrences in `template_args`.
3. DFBs in `dfb_dependencies`.

Each occurrence receives its position in that returned value sequence as its IR
index; the sequence does not describe execution. It preserves occurrences
rather than uniquing SSA values. Operand adaptation can map distinct source
operands to the same DFB, but each occurrence retains its own access contract.

Each Python effect explicitly names the affected DFB. For example,
`ttl.DFBEffect.wait(source, tiles=1)` means that the external function executes
a one-tile wait on `source`. The frontend resolves `source` to its dependency
occurrence and stores that occurrence index in `DFBProtocolEffectAttr`. The IR
attribute therefore contains a typed action, a dependency index, and a positive
static tile count no greater than the DFB capacity; the dependency index is not
an execution position.

The following call declares two descriptor dependencies and four external
protocol actions:

```python
ttl.call_extern_func(
    HEADER,
    "external_copy",
    template_args=[
        ttl.dfb_descriptor(source),
        ttl.dfb_descriptor(destination),
    ],
    dfb_effects=[
        ttl.DFBEffect.reserve(destination, tiles=1),
        ttl.DFBEffect.wait(source, tiles=1),
        ttl.DFBEffect.push(destination, tiles=1),
        ttl.DFBEffect.pop(source, tiles=1),
    ],
    kernel=ttl.KernelKind.DATA_MOVEMENT,
)
```

`dfb_effects` is one call-wide execution sequence. List position specifies the
order in which the external C++ executes protocol actions, including actions on
different DFBs. Different DFBs do not share an order position; their actions
occupy distinct positions in the same sequence. The call above produces this
dependency sequence and effect sequence:

```text
Sequence returned by getDFBDependencyOperands():

  index 0: source
  index 1: destination

External-call execution sequence:

  call entry
      |
      v
  [0] reserve destination, 1 tile  (dependency 1)
      |
      v
  [1] wait source, 1 tile          (dependency 0)
      |
      v
      copy source -> destination   (ordinary C++ storage access)
      |
      v
  [2] push destination, 1 tile     (dependency 1)
      |
      v
  [3] pop source, 1 tile           (dependency 0)
      |
      v
    return

Per-DFB protocol subsequences:

  source:      wait [1] ----------------------------> pop [3]
  destination: reserve [0] --------> push [2]
```

The global positions preserve the cross-DFB relation: destination is reserved
before source is popped, so their lifetimes overlap. The compiler uses this
relation when deciding whether physical-index reuse is valid.

The frontend resolves static tile and repeat expressions, recursively expands
`ttl.DFBEffect.repeat`, and rejects an empty repeat body or a flattened sequence
longer than 4096 actions before creating IR. A repeat inserts each copy of its
body into the same call-wide sequence. The 4096-action bound matches other
bounded static event enumerations in the compiler and limits both frontend
materialization and downstream per-effect analysis. It is not a hardware
limit. A repeat has no IR or runtime representation.

`ttl.opaque_call` implements `DFBAccessOpInterface`, which supplies four facts
without exposing the operation's operand-segment representation to analyses:

| Interface fact | Meaning |
| --- | --- |
| DFB dependencies | Every statically declared storage-access occurrence. |
| Protocol effects | Synchronous reserve, push, wait, and pop actions in call execution order. |
| DFB index operands | DFBs whose finalized physical indices reach external C++. |
| Unknown access | The call may access unlisted user-managed DFBs. |

An effect summary describes actions performed by external C++; it does not
create executable TTL operations. Every listed action must execute on every
execution of the call and complete before return. Conditional external actions
require corresponding TTL control flow around a call with an unconditional
summary. The effect list does not state when associated hardware interface work
completes; the declared protocol terminal or a synchronized reset must complete
that work before storage reuse.

An occurrence with no effect is a possible read and write beginning at call
entry. A synchronized reset ordered after the call through the same
participating logical kernel terminates a named opaque access and canonicalizes
its protocol state. The reset implementation must complete earlier interface
work before publishing arrival. Ordinary storage accesses between summarized
acquisitions and releases remain inside the corresponding lifetime. A partial
effect sequence is valid metadata but cannot prove a bounded lifecycle for that
DFB. When the same DFB has multiple dependency occurrences, every occurrence
requires effects to eliminate the opaque access without a reset. For allocation,
`unknown_dfb_access` conservatively adds the call as an opaque occurrence on
every user-managed DFB, including listed DFBs, in each affected allocation
scope. The compiler does not infer facts from the callee name, header, emitted
C++, or integer DFB identity.

DFB ownership, synchronization insertion, SPSC verification, and physical
allocation consume this common interface. Their use of opaque and effectful
accesses is described in
[Dataflow Buffer Management](DFBManagement.md#external-calls).

## Typed DFB descriptors

A descriptor supplies the finalized physical allocation properties required by
an external DFB protocol. The two descriptor arguments in the `external_copy`
call above lower to a C++ template invocation equivalent to:

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
  static_assert(Source::pages_per_block == Destination::pages_per_block);
  static_assert(Source::page_size_bytes == Destination::page_size_bytes);
  cb_reserve_back(Destination::index, Destination::pages_per_block);
  cb_wait_front(Source::index, Source::pages_per_block);

  auto *source_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
      get_read_ptr(Source::index));
  auto *destination_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
      get_write_ptr(Destination::index));
  constexpr uint32_t total_words =
      Source::pages_per_block * Source::page_size_bytes / sizeof(uint32_t);
  for (uint32_t word_index = 0; word_index < total_words; ++word_index) {
    destination_ptr[word_index] = source_ptr[word_index];
  }

  cb_push_back(Destination::index, Destination::pages_per_block);
  cb_pop_front(Source::index, Source::pages_per_block);
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
    dfb_dependencies=[source],
    kernel=ttl.KernelKind.DATA_MOVEMENT,
)
```

The dependency-only operand is required because an integer index argument does
not declare that the external function accesses DFB storage. It does not add a
C++ function argument.

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

The Python AST emits ordered typed attributes for static values and separate
operand segments for template, dependency-only, and function-argument DFBs.
Dependency occurrences, protocol effects, and unknown access remain in TTL for
analysis and verification. TTL to TTKernel conversion resolves DFB indices and
materializes descriptor metadata before the DFB type loses its block geometry;
it discards dependency-only and effect metadata because those facts do not
alter the C++ call. TTKernel to EmitC resolves constants and descriptor types,
and C++ emission inserts the required prelude and header during its existing
operation scan.

See `examples/external_dfb_descriptors.py` for two external calls surrounded by
visible TTL protocol operations. Descriptor operands preserve each DFB's
compile-time metadata but do not summarize storage behavior, so the two opaque
result DFBs remain distinct. See
`test/python/call_extern_func_dfb_effects.py` for dependency-only operands,
ordered effect expansion, unknown access, and generated-C++ invariance.
