# Operation Runtime Resources

`@ttl.operation` accepts a `runtime_resource_factory` callback for resources
that depend on the current invocation. The callback provides caller-created
program semaphores, per-logical-kernel runtime arguments and definitions, and
host objects that must remain alive through execution.

The callback has the following keyword-only contract:

```python
def make_resources(*, tensors, core_ranges, first_free_semaphore_id):
    ...
```

- `tensors` contains the current invocation tensors.
- `core_ranges` contains the operation worker cores.
- `first_free_semaphore_id` is the first ID after compiler-managed semaphores.

The callback executes once for every operation invocation and returns a
`ttl.ProgramRuntimeResources`. The callback result is not cached.

## Typed Records

All resource records are frozen, and all collection fields are tuples.

| Record | Purpose |
| --- | --- |
| `ProgramRuntimeResources` | Contains caller semaphore descriptors, logical-kernel resources, external fabric bindings, and retained owners. |
| `KernelRuntimeResources` | Selects one logical kernel and supplies per-core runtime arguments and JIT definitions. |
| `CoreRuntimeArgs` | Associates one ordered integer vector with one worker coordinate. |
| `KernelDefine` | Associates one definition name with its string value. |
| `FabricManagerClaim` | Identifies one external fabric manager and its logical kernel. |
| `FabricConnectionBinding` | Associates one captured manager claim with its connection requirements and ABI identity. |
| `FabricConnectionRequirement` | Reserves one fixed forwarding link for specified logical devices and worker nodes. |

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

## External Fabric Managers

An opaque external kernel may own routing-plane connections that must not
overlap compiler-generated managers. A captured `FabricManagerClaim` records
that ownership without exposing the external kernel's runtime-argument ABI:

```python
external_sender = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
external_manager = ttl.FabricManagerClaim("external", kernel=external_sender)

def sender(input_tensor, output_tensor):
    ttl.call_extern_func(
        HEADER,
        "open_connections",
        func_args=[],
        kernel=external_sender,
        fabric_manager_effects=(external_manager.acquire(),),
    )
    ttl.call_extern_func(
        HEADER,
        "use_connections",
        func_args=[input_tensor, output_tensor],
        kernel=external_sender,
        fabric_manager_effects=(external_manager.use(),),
    )
    ttl.call_extern_func(
        HEADER,
        "close_connections",
        func_args=[],
        kernel=external_sender,
        fabric_manager_effects=(external_manager.release(),),
    )
```

Use `ttl.PIPE_SOURCE_KERNEL` as the claim kernel when the external manager must
execute on the compiler-owned PipeNet source kernel.

Ownership begins at entry to the acquire call because that opaque call may open
connections. It ends after the release call returns. Every `use()` must occur
strictly between one acquire and release. `scoped()` declares one opaque call
that acquires, uses, and releases the manager. Acquire, use, and release
effects must occur in the selected logical kernel's straight-line entry block.
A scoped effect may occur in structured control flow when the compiler proves
its exact launch-node domain.

An expand-only operation forwards captured claims to the final grid-bearing
operation. Acquire, use, and release effects may therefore reside in separate
composed helpers. The claim binds once when the final operation is registered;
reusing the same claim in two independent grid-bearing operations is invalid.

The runtime resource factory supplies one `FabricConnectionBinding` for every
captured claim. Its requirements must cover every active logical-device and
worker-node instance of the selected kernel. A conditional scoped effect uses
its proven launch-node domain instead of the complete executable descriptor;
a proven empty domain requires no worker binding. `fixed_link_index` is an
external ABI constraint, not a preference; target binding rejects it when the
active control plane does not expose that link for the destination. The
compiler places external and generated manager intervals in one interference
graph, validates every link assignment before descriptor mutation, and
reserves the external link without interpreting or modifying external runtime
arguments.

The claim identity, `abi_identity`, logical endpoints, worker nodes, and fixed
links participate in program-cache identity. Objects in the binding's
`lifetimes` tuple remain referenced through execution.

## Validation and Specialization

Runtime resource planning validates the complete factory result before TTNN
descriptor construction. Invalid resources raise an exception without
materializing a partial program. Validation includes the following rules:

- Runtime argument coordinates and semaphore ranges must be inside the
  operation worker range.
- Caller semaphore IDs must be unique and greater than or equal to
  `first_free_semaphore_id`.
- Definition names and values must be strings, and definitions for one logical
  kernel must not conflict.
- Runtime argument vectors for one descriptor must have equal lengths.
Core specialization can produce several descriptors for one logical kernel.
Definitions apply to every descriptor for that identity. Runtime arguments are
partitioned by coordinate, and every coordinate must match exactly one
descriptor.

## Program Cache and Lifetimes

Resource structure participates in program-cache identity. The identity
includes logical kernels, descriptor coordinates, definitions, runtime-vector
lengths, caller semaphore properties, and external fabric binding structure.
Runtime values, tensor addresses, and lifetime object identities are excluded,
so cached programs accept new invocation values.

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
materializer as normal execution. It does not serialize live owners or create
replacement topology resources.

Operation runtime resources are a hardware execution interface. The simulator
does not model TTNN program descriptors or per-core kernel runtime arguments
and rejects `runtime_resource_factory`.
