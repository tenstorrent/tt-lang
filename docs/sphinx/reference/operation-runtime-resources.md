# Operation Runtime Resources

TT-Metal programs separate reusable structure from dispatch values. Kernel
definitions, program semaphore layout, and runtime-argument schema determine
which program can be reused; per-core runtime-argument words may change for
each dispatch. Execution may also require host or device objects to remain
alive while the program runs.

The `runtime_resource_factory` callback supplies caller-defined program
semaphores and JIT definitions, per-core dispatch words, and host owners for a
TT-Lang operation. It runs once for each device execution, before the runner
constructs program descriptors and after the current tensors, launch range,
and compiler-reserved semaphore IDs are known. Resources select logical kernels
rather than generated descriptor indices, which can change with target
selection and core specialization.

The callback has the following keyword-only contract:

```python
def make_resources(*, tensors, core_ranges, first_free_semaphore_id):
    ...
```

- `tensors` contains the current invocation tensors.
- `core_ranges` contains the operation worker cores.
- `first_free_semaphore_id` is the first ID after compiler-managed semaphores.

The callback executes once for every device execution and returns a
`ttl.ProgramRuntimeResources`. The callback result is not cached. Its structural
fingerprint may select an existing cached TT-Metal program, while its current
runtime-argument words are supplied for that dispatch.

## Typed Records

All resource records are frozen, and all collection fields are tuples.

| Record | Purpose |
| --- | --- |
| `ProgramRuntimeResources` | Contains caller semaphore descriptors, logical-kernel resources, and retained owners. |
| `KernelRuntimeResources` | Selects one logical kernel and supplies per-core runtime arguments and JIT definitions. |
| `CoreRuntimeArgs` | Associates one ordered integer vector with one worker coordinate. |
| `KernelDefine` | Associates one definition name with its string value. |

`KernelDefine` is compile-affecting program structure even though the factory
returns it for each execution. `CoreRuntimeArgs.values` contains the dispatch
words that may change while the cached program is reused.

The following example creates one caller semaphore and configures an
operation-owned data-movement kernel:

```python
def make_collective(runtime_owner):
    sender = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)

    def make_resources(*, tensors, core_ranges, first_free_semaphore_id):
        semaphore = ttnn.SemaphoreDescriptor(
            first_free_semaphore_id,
            core_ranges=core_ranges,
            initial_value=0,
        )
        return ttl.ProgramRuntimeResources(
            semaphore_descriptors=(semaphore,),
            kernel_resources=(
                ttl.KernelRuntimeResources(
                    kernel=sender,
                    runtime_args=(
                        ttl.CoreRuntimeArgs(
                            ttnn.CoreCoord(0, 0),
                            (first_free_semaphore_id, 0),
                        ),
                        ttl.CoreRuntimeArgs(
                            ttnn.CoreCoord(1, 0),
                            (first_free_semaphore_id, 1),
                        ),
                    ),
                    defines=(ttl.KernelDefine("FABRIC_2D", "1"),),
                ),
            ),
            lifetimes=(runtime_owner, semaphore),
        )

    @ttl.operation(grid=(2, 1), runtime_resource_factory=make_resources)
    def collective(input_tensor, output_tensor):
        ttl.call_extern_func(
            HEADER,
            "collective_sender",
            func_args=[input_tensor, output_tensor],
            kernel=sender,
        )

    return collective
```

`KernelKind.COMPUTE` and `KernelKind.DATA_MOVEMENT` select canonical kernels.
A captured `Kernel` identifies an operation-owned kernel when an operation has
multiple kernels of one kind. The operation and its resource factory must
capture the same `Kernel` object. The [external functions
reference](external-functions.md) describes logical-kernel selection.

## Validation and Specialization

Runtime resource planning validates the complete factory result before the
runner constructs a TTNN `KernelDescriptor` or `ProgramDescriptor`. Invalid
resources raise an exception without materializing a partial program.
Validation includes the following rules:

- The result and every nested resource use the documented record and tuple
  types.
- Every explicit `Kernel` is bound to the executing operation and identifies
  an emitted logical kernel.
- Runtime argument coordinates and semaphore ranges must be inside the
  operation worker range.
- Each runtime argument coordinate occurs at most once per logical kernel, and
  its values are integer-indexable non-boolean objects.
- Caller semaphore IDs must be unique and greater than or equal to
  `first_free_semaphore_id`.
- Each logical kernel may have at most one `KernelRuntimeResources` entry.
- Definition names and values must be strings, and each definition name must
  occur at most once in that entry.

Core specialization can produce several descriptors for one logical kernel.
Definitions apply to every descriptor for that identity. Runtime arguments are
partitioned by coordinate, and every coordinate must match exactly one
descriptor.

## Program Cache and Lifetimes

Resource structure participates in program-cache identity. The identity
includes logical destinations, descriptor coordinates, definitions,
runtime-argument coordinates and vector lengths, and caller semaphore
properties. Runtime-argument words, tensor addresses, and lifetime object
identities are excluded, so cached programs accept new dispatch values.

Objects in `ProgramRuntimeResources.lifetimes` remain referenced through
execution. A successful invocation replaces the retained owner tuple. A failed
invocation preserves the previous valid owners.

## Emitted Runners and Simulator

An emitted runner for a resource-aware operation requires the factory on every
call:

```python
runner.run(
    tensors,
    runtime_resource_factory=make_resources,
    device=device,
)
```

The runner reconstructs logical-kernel identities and uses the same planner and
materializer as decorated operation execution. It does not serialize live
owners or dispatch values; the caller supplies them through the factory.

Operation runtime resources are a hardware execution interface. The simulator
does not model TTNN program descriptors or per-core kernel runtime arguments
and rejects `runtime_resource_factory`.
