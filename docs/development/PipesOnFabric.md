# Pipes on Fabric

This document describes the hardware capabilities, compiler design, software
architecture, implementation, and validation of cross-device PipeNets in
tt-lang. It complements [PipeNets](PipeNets.md), which defines the shared pipe
semantics used by both local NoC and fabric transports.

## System overview

```text
  Python operation
  DeviceDomain + TransferGraph + graph PipeNet
                         |
                         v
  +-----------------------------------------------------------+
  | Logical transfer semantics                                |
  | source/destination relation and DFB ownership             |
  | synchronization protocol                                  |
  +-----------------------------------------------------------+
                         |
                         v
  +-----------------------------------------------------------+
  | Target route resolution                                   |
  | FabricNodeIds, links, directions, legal packet routes     |
  +-----------------------------------------------------------+
             |                              |
             | one proven direct segment    | multiple segments
             v                              v
  +--------------------------+   +-----------------------------+
  | Source TENSIX node       |   | Source TENSIX node          |
  | fabric write + atomic    |   | fabric write + atomic       |
  +--------------------------+   +-----------------------------+
             |                              |
             | fabric                       v
             |                   +-----------------------------+
             |                   | Intermediate device         |
             |                   | receive DFB + semaphore     |
             |                   | fabric write + atomic       |
             |                   +-----------------------------+
             |                              |
             v                              v
  +-----------------------------------------------------------+
  | Destination device                                        |
  | completion wait -> destination DFB consumption            |
  +-----------------------------------------------------------+
```

High-level TTL records the logical transfer only. Target resolution decides
whether the transfer uses one direct packet route or an explicit
receive-and-forward segment sequence.

## Hardware capabilities

### TENSIX nodes and dataflow buffers

A TENSIX node contains five Baby RISC-V processors, local SRAM, hardware
semaphores, and dataflow buffers (DFBs). Data movement kernels execute on the
node's data movement processors. Compute kernels execute on its compute
processors. A DFB is an L1-resident FIFO used to transfer tiles between those
threads.

Pipe payload storage remains receiver-owned for both local and fabric
transfers. The receiver reserves a DFB block before the sender writes into it.
The fabric does not provide a hidden payload buffer with PipeNet semantics.
Fabric routers and packet buffers transport data between devices, while the
destination address still identifies storage in a receiver node's L1.

### On-device NoC and inter-device fabric

The NoC transfers data between nodes and memory controllers on one device. A
NoC unicast address contains the translated destination node coordinates and
the destination L1 address. Existing PipeNet lowering uses NoC writes and NoC
semaphore increments for transfers whose endpoints are on the same device.

The TT-Metal fabric transfers packets between devices. A fabric packet
contains two distinct destinations:

- a chip route selects the destination device or the sequence of fabric
  routers;
- a NoC command selects the destination node and L1 address after the packet
  reaches the destination device.

These destinations must not be conflated. A logical `DeviceRef` determines
which device participates in a transfer. Immediately before submitting a
compiled operation, the host runtime constructs per-device `ProgramDescriptor`
runtime arguments and a `MeshProgramDescriptor`. During this execution-setup
stage, called *host runtime route binding* below, it resolves each logical
device to a `FabricNodeId` and queries the control plane owned by the active
`MeshDevice` runtime context for a legal route. Binding completes before
`ttnn.generic_op(...)` submits the program; it does not execute in a device
kernel. The packet's NoC command is built separately from the destination node
coordinates and receiver DFB address.

### Routing-plane connections

Generated data movement kernels use TT-Metal's
`tt::tt_fabric::RoutingPlaneConnectionManager`. Host code configures each
connection with `ttnn.setup_routing_plane_connection(...)`. That call:

- selects a forwarding direction and fabric link;
- allocates the connection semaphores in the program descriptor;
- adds the kernel defines required by the selected fabric API;
- returns the runtime arguments consumed by
  `RoutingPlaneConnectionManager::build_from_args()`.

The kernel opens the required connections before sending, obtains the sender
associated with a connection slot, and closes all opened connections before
returning. Connection selection and packet routing are separate operations. A
connection chooses a router injection direction and link. The packet header
still identifies how far or to which device the packet travels.

### Packet routing

TT-Metal currently exposes different packet-header routing behavior for linear
and mesh fabrics.

For a linear fabric, the packet header uses a hop count:

```cpp
packet_header->to_chip_unicast(num_hops);
```

Under `FABRIC_1D`, `fabric_set_unicast_route()` does not populate this field.
Omitting `to_chip_unicast()` allows connection setup and packet submission to
complete, but the destination does not receive the packet.

For a 2D fabric, routing-plane setup supplies the source and destination mesh
metadata used by:

```cpp
fabric_set_unicast_route(connection_manager, packet_header, connection_slot);
```

The selected operation is a target-runtime decision. Neither the fabric mode
nor either route encoding is part of the TTL domain or transfer-graph model.

### Payload and completion ordering

The current fabric transport uses the fused packet command
`to_noc_fused_unicast_write_atomic_inc`. The packet writes the payload to the
destination DFB and increments the receiver-completion semaphore as one fabric
command. The receiver waits for that semaphore before consuming the DFB block.

The sender follows TT-Metal's packet lifecycle:

1. Reset and allocate a packet header.
2. Encode the chip route.
3. Encode the destination NoC write and completion increment.
4. Wait for an empty fabric write slot.
5. Submit the payload without the header.
6. Flush the header with a blocking operation.
7. Close the opened connections.

The target runtime reports the maximum fabric payload size. Code generation
must not embed a device-specific limit. The P300 linear-router baseline used a
4,096-byte payload, below its observed 4,416-byte maximum. Larger PipeNet
payloads require target-aware packetization, with the completion increment in
the final packet.

### Physical and logical device arrangements

A `MeshDevice` logical arrangement is not necessarily the physical fabric
arrangement. The current P300_X2 control plane reports a `2x2` mesh extent.
General collective tests query that extent and construct the logical
`DeviceDomain` from it instead of encoding a fixed extent. The extent defines
logical membership and row-major ordering; it does not define fabric
adjacency. Logical coordinate distance is therefore not a fabric hop count.

The same distinction applies to `FabricNodeId::chip_id`. It is an identifier
within a physical mesh, not a general distance metric. A target resolver must
consult TT-Metal's control-plane description to determine whether a transfer
is directly routable and how to encode it.

## Design

### Design goals

Fabric pipes extend PipeNet communication across devices without making the
TTL programming model depend on a current Tenstorrent system topology. The
design has the following invariants:

- `DeviceDomain`, `DomainMap`, `DeviceRef`, and `TransferGraph` contain no
  fabric mode, physical mesh identifier, route direction, link index, hop
  count, packet limit, or NoC selection.
- `DeviceRef` identifies a logical member of a `DeviceDomain`. It is not a
  physical device identifier.
- PipeNet protocol planning is shared by local NoC and fabric transports.
- PipeNet identifiers preserve semantic identity and never select physical
  semaphore ids or runtime-argument indices.
- Host runtime route binding resolves logical devices, physical routes,
  transport limits, and connection metadata while constructing program
  descriptors before submission.
- Runtime communication state scales with local live degree and queue depth,
  not the total domain or transfer-graph size.
- Source programs and high-level TTL IR remain valid when a target uses a
  different topology or routing API.

### Programming model

`DeviceDomain` is a logical index set of devices. It follows the separation
used by Chapel domains and locales: the domain defines membership, while a
target-specific mapping determines physical placement.

`DeviceRef` identifies one member of a device domain. `DomainMap` describes
ownership and distribution over a domain. `TransferGraph` describes the
logical communication relation. `PipeNet` applies the existing pipe protocol
to that relation.

`DeviceDomain.current_index()` returns the zero-based row-major order of the
current logical device. Pipe callback identities expose source and destination
indices using the same ordering. These indices support distributed tensor
offsets without exposing physical device identifiers or route coordinates.

For example, a point-to-point transfer graph records only its logical edge:

```python
devices = ttl.DeviceDomain((1, 4))
transfers = ttl.TransferGraph.edges(
    devices,
    edges=[((0, 0), (0, 3))],
)
net = ttl.PipeNet(graph=transfers)
```

The graph does not state whether the target uses a line, ring, torus, mesh, or
another interconnect. It also does not require `(0, 0)` and `(0, 3)` to be one
hardware packet apart.

Structured transfers share common domain and component properties through
`StructuredTransfer`. Current derived forms include axis-neighbor, gather,
and multicast relations. Additional collectives should add semantic transfer
forms such as scatter and all-to-all without adding target topology fields.

### Shared pipe protocol

Local and fabric transfers use the same logical protocol:

- the receiver owns and reserves the destination DFB block;
- the receiver publishes readiness or participates in a proven capacity
  protocol;
- the sender writes into the receiver-owned block;
- the receiver waits for a completion signal before consuming the block;
- source and destination roles are restricted by PipeNet guards;
- DFB reserve, wait, push, and pop lifetimes remain visible to shared compiler
  analyses.

The transport emitter maps that protocol onto either NoC or fabric
operations. It does not redefine PipeNet semantics.

### Synchronization storage

The resource planner selects synchronization storage independently from
PipeNet identity. Local transfers use densely allocated hardware semaphore ids
when every access remains on one device. A completion counter targeted by a
fabric atomic uses a host-created `GlobalSemaphore`; a local semaphore id
cannot identify an object on another device.

`GlobalSemaphore` provides a common L1 address on the selected TENSIX nodes of
every device. The sender receives that address as a common runtime argument and
combines it with the destination node coordinates to form the remote NoC
address. The receiver uses the same runtime argument as the local address for
its completion wait. Route metadata and synchronization storage are therefore
independent: changing the selected route does not change PipeNet semantics or
counter identity.

### Route resolution requirements

A resolved transport plan converts one logical transfer edge into a route with
one or more transport segments. Each segment records:

- the local and remote `FabricNodeId` endpoints;
- the connection slot and link selection;
- the packet-header route encoding;
- the source and destination TENSIX nodes;
- the source and destination L1 addresses;
- the payload size and packetization constraints;
- the synchronization objects used by that segment.

A target may use one direct segment only when its control plane proves that
the destination is representable by one packet-header route. A route requiring
turns, forwarding nodes, or different transport mechanisms must be represented
as multiple segments.

This rule is important for linear fabric. A direct hop count is valid only
when the source and destination lie on one physical fabric axis. A logical
edge that requires a turn cannot be encoded by increasing the hop count.

### Multi-segment forwarding

A validated segmented transport decomposes a logical source-to-destination
transfer into adjacent logical devices. Each intermediate device receives into
a local DFB, waits for completion, then sends the payload to the next device.
Every P300 linear-fabric packet uses `to_chip_unicast(1)`.

tt-lang should preserve those generated-kernel properties while obtaining the
segment sequence from the target control plane. The compiler representation
remains target-independent:

- the high-level transfer stays one logical edge;
- target route resolution decides the segment sequence;
- intermediate forwarding is introduced after logical transfer analysis;
- targets that support direct multi-hop or mesh routing may select one
  segment;
- targets that require forwarding may select several segments.

Intermediate forwarding requires explicit DFB and synchronization resources
on each forwarding device. Those resources belong to the resolved transport
plan, not the source `TransferGraph`.

### Collective communication

Collectives are transfer relations plus local computation. They should reuse
the same route resolver and fabric transport emitter instead of implementing
an unrelated communication subsystem.

The fabric pytest suite is intended to cover:

- point-to-point;
- broadcast;
- reduce-to-root;
- all-gather;
- reduce-scatter;
- all-reduce;
- all-to-all.

Each collective may select a target-specific algorithm after the logical
relation is known. Ring, tree, and direct-exchange algorithms are lowering
strategies, not high-level domain properties.

## Software architecture

### Compilation flow

Cross-device pipe processing has separate compilation and execution-setup
stages:

```text
Compilation:
Python DeviceDomain, DeviceRef, TransferGraph, and PipeNet
  -> TTL device-domain and transfer attributes
  -> Pipe Transfer IR and shared PipeNet resource planning
  -> logical fabric-route records attached to generated kernels
  -> TTKernel routing-plane operations
  -> EmitC calls into tt::tt_fabric

Host execution setup for each invocation:
compiled kernel route records + active MeshDevice
  -> host runtime route binding
  -> FabricNodeId and MeshDevice-scoped control-plane route queries
  -> per-device ProgramDescriptor runtime arguments
  -> MeshProgramDescriptor construction
  -> TTNN MeshProgramDescriptor execution
```

Logical transfer analysis remains independent of physical topology. Concrete
fabric configuration, link selection, and packet-route values first enter
during host runtime route binding, after kernel generation and before
`ttnn.generic_op(...)` submission. Generated kernels consume those values as
runtime arguments.

### Frontend and TTL IR

The Python domain model is implemented in `python/ttl/domains.py`. The AST
lowering in `python/ttl/_src/ttl_ast.py` converts logical domain members and
transfer edges into TTL attributes.

The TTL dialect defines:

- `DeviceDomainComponentAttr` for one named logical index-space component;
- `DeviceDomainAttr` for a product of components;
- `DeviceRefAttr` for logical coordinates within that domain;
- `DeviceRangeAttr` for a logical device range;
- `TransferEdgeAttr` for one logical transfer relation;
- `DeviceTransferAttr` for binding a logical device edge to a node-level
  pipe.
- `CurrentDeviceIndexOp` for the current member's row-major logical index.

These attributes contain no target route fields. Their verifiers check domain
membership, coordinate rank, and transfer structure.

### Pipe lowering

`lib/Dialect/TTL/Transforms/PipeLowering.cpp` builds one fabric-route plan for
the module before lowering individual sends. The current POC records the
logical local device, logical remote device, and source node set for each
route. It assigns each fabric send a stable route index.

The shared `PipeTransportEmitter` interface separates PipeNet protocol
planning from transport emission. `NocPipeTransportEmitter` emits existing
same-device NoC operations. `FabricPipeTransportEmitter` emits routing-plane
atomics and fused payload-write-plus-completion operations.

The resource planner resolves each synchronization counter to an L1 address
before invoking the transport emitter. The emitter consumes that address and
does not interpret a PipeNet id as a semaphore id. A PipeNet containing a
cross-device transfer allocates its completion counter from the global
semaphore namespace; local-only PipeNets allocate completion counters densely
from the local namespace.

Current fabric lowering requires computed receiver DFB addresses. The sender
uses the destination DFB base address supplied by the host runtime and builds
the remote NoC address from the destination node coordinates. Receiver-
published address-table fallback remains unsupported for fabric transfers.

### TTKernel representation

The POC adds TTKernel operations for the routing-plane manager lifecycle and
packet submission:

- create a `RoutingPlaneConnectionManager` value;
- open connections from a runtime argument block;
- submit a remote atomic increment;
- submit a fused payload write and remote atomic increment;
- close the opened connections.

The send operations take both a connection index and a target-provided
`chipRoute`. The connection index selects a manager slot. `chipRoute` contains
the packet-header value required by the selected target configuration.

These operations do not use tt-mlir's experimental fabric operations. Those
operations consume a different runtime argument layout produced by the
flatbuffer runtime. tt-lang executes through `ttnn.generic_op` and
`MeshProgramDescriptor`, so its TTKernel operations model the
`RoutingPlaneConnectionManager` ABI directly.

### Comparison with tt-mlir's experimental manager

Both implementations ultimately use TT-Metal's
`tt::tt_fabric::RoutingPlaneConnectionManager`. The distinction is where
topology interpretation and route selection occur. At tt-mlir commit
[`7a1e911f83`](https://github.com/tenstorrent/tt-mlir/commit/7a1e911f83ff5d380703309732d2a91e70104b07),
its
[`experimental::FabricConnectionManager`](https://github.com/tenstorrent/tt-mlir/blob/7a1e911f83ff5d380703309732d2a91e70104b07/include/ttmlir/Target/TTKernel/LLKs/experimental_fabric_api.h#L52-L90)
wraps the TT-Metal manager with topology state, packet-header ownership, and
initialization state.

The comparison uses these execution-location markers:

- `[C]`: compiler work performed before invocation;
- `[H]`: host runtime work performed while constructing program descriptors,
  before `ttnn.generic_op(...)` submission;
- `[D]`: worker-kernel work performed on a TENSIX node.

Route-decision work occurs at different locations and frequencies:

```text
tt-lang
  [C] Record each logical source/destination relation and assign a route index.
  [H] Query the control plane once per distinct source/destination pair; cache
      the result while the MeshDevice and fabric configuration remain unchanged.
  [H] Write the selected connection index and packet-route value into runtime args.
  [D] Once per kernel invocation: open the configured connections.
  [D] Once per fabric operation: index the selected connection, construct the
      packet command, and submit it.

tt-mlir experimental API
  [H] Serialize TopologyInfo and connection descriptors into runtime args.
  [D] Once per kernel invocation: parse TopologyInfo and open the connections.
  [D] Once per fabric operation: reconstruct logical positions, calculate the
      direction and hops, search connection tags, construct the packet command,
      and submit it.
```

These markers describe route selection and packet construction. After packet
submission, TT-Metal's fabric routers perform the actual packet forwarding for
both approaches.

| Concern | tt-lang | tt-mlir experimental API |
| --- | --- | --- |
| C++ object | `[D]` Uses `tt::tt_fabric::RoutingPlaneConnectionManager` directly. | `[D]` Wraps the same manager in `experimental::FabricConnectionManager`. |
| Route resolution | `[H]` Queries the active TT-Metal control plane and supplies a connection index and target-specific `chipRoute`. | `[D]` Derives an outgoing direction and hop encoding from destination mesh and device identifiers for each fabric operation. |
| Transfer operands | `[C]` Routing-plane TTKernel operations represent a resolved connection index and route value. `[D]` The generated kernel reads both from runtime arguments. | `[C]` [Fabric TTKernel operations](https://github.com/tenstorrent/tt-mlir/blob/7a1e911f83ff5d380703309732d2a91e70104b07/include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td#L4227-L4320) represent destination mesh and device identifiers. `[D]` The wrapper resolves them. |
| Connection selection | `[D]` Generated C++ indexes the selected manager slot directly. | `[D]` The wrapper [searches active connection tags](https://github.com/tenstorrent/tt-mlir/blob/7a1e911f83ff5d380703309732d2a91e70104b07/include/ttmlir/Target/TTKernel/LLKs/experimental_fabric_api.h#L38-L76) for each fabric operation. |
| Runtime arguments | `[C]` The compiler assigns an explicit base for the connection descriptor block. `[H]` The binder appends control-plane-resolved route and connection values. `[D]` Connection setup reads them once per kernel invocation. | `[H]` The runtime serializes topology and connection descriptors into a [fixed fabric argument block](https://github.com/tenstorrent/tt-mlir/blob/7a1e911f83ff5d380703309732d2a91e70104b07/include/ttmlir/Target/TTKernel/LLKs/experimental_fabric_api.h#L105-L135). `[D]` Setup parses that block once per kernel invocation. |
| Topology model | `[C]` Logical TTL domains contain no fabric topology constants. `[H]` Host runtime route binding queries TT-Metal for the active topology's route. | `[H]` The runtime supplies a topology descriptor encoding [two dimensions, four directions, a 32-device limit, and line/ring/mesh/torus categories](https://github.com/tenstorrent/tt-mlir/blob/7a1e911f83ff5d380703309732d2a91e70104b07/include/ttmlir/Target/TTKernel/LLKs/experimental_fabric_topology_info.h#L14-L35). `[D]` Kernel helpers interpret it. |
| Current operation coverage | Provides the unicast atomic and fused write-plus-atomic operations required by fabric PipeNets. | Also provides arbitrary-length packetization and multicast write and semaphore helpers. |
| Per-operation kernel work | `[D]` Indexes a pre-resolved connection and constructs the packet command. | `[D]` Performs [logical-position route calculation](https://github.com/tenstorrent/tt-mlir/blob/7a1e911f83ff5d380703309732d2a91e70104b07/include/ttmlir/Target/TTKernel/LLKs/experimental_fabric_1d_routing.h#L36-L108) and connection-tag lookup before constructing the packet command. |

The tt-lang representation is intentionally lower-level at the TTKernel
boundary. Logical destinations remain available in TTL until host runtime route
binding, but generated kernels receive only values already validated against
the active control plane. This avoids embedding the current architecture's
topology categories in high-level domain attributes. The tt-mlir API currently
provides broader packet and multicast helpers; equivalent tt-lang functionality
must preserve host-side route resolution rather than reintroducing an in-kernel
topology model.

The design comparison has the following implications:

| Goal | tt-lang | tt-mlir experimental API |
| --- | --- | --- |
| Architecture portability | `[C]` Logical domains and transfer graphs contain no topology categories or link counts. `[H]` While constructing each invocation's program descriptors, the host runtime queries the active TT-Metal control plane and writes the resolved connection and packet-route values into kernel runtime arguments. A new route encoding requires corresponding host runtime route binding and TTKernel/EmitC lowering support. | `[H]` The runtime serializes explicit line, ring, mesh, and torus models with fixed dimension and device limits. `[D]` The kernel wrapper interprets that model. Supporting a new topology requires changes to the topology descriptor, runtime serialization, and kernel routing helpers. |
| Extensibility | Logical transfers, route planning, connection reuse, and packet emission are separate components. New route scoring or packet operations do not change domain semantics, but the compiler/runtime argument contract must track TT-Metal API changes. | The wrapper gives TTKernel operations a compact destination-oriented API and already provides broader packetization and multicast helpers. Extending its topology model also increases on-device wrapper state and routing logic. |
| Kernel performance | `[H]` Route queries or cache lookups occur while the host constructs an invocation's program descriptors, before program submission. `[D]` Kernels directly index reused connection slots without calculating a logical route or searching connection tags for each fabric operation. Current planning selects the control plane's first available link and does not yet score contention or workload traffic. | `[D]` Destination-oriented operations support runtime-selected destinations within the encoded topology. Each fabric operation reconstructs source and destination logical positions, performs topology-dependent hop calculation, and linearly searches active connection tags. This work is potentially material for small or frequent transfers. |
| Route optimization | `[H]` Host planning can evaluate legal control-plane routes using global program information and can cache the selected plan. | `[D]` For every fabric operation, kernel helpers calculate the direction and hop encoding from local topology and destination values. This limits access to program-wide traffic information. |
| Maturity | Generated C++ directly exposes the operations needed to converge on optimized routing-plane kernel sequences. Current packet and multicast coverage is narrower, and comparative performance has not yet been measured. | The experimental API currently covers more unicast, multicast, arbitrary-length write, and semaphore operations, but embeds more current-architecture policy in the kernel support library. |

The tt-mlir routing helpers are `FORCE_INLINE`, which removes function-call
overhead but does not generally remove the route calculation. `TopologyInfo`
is populated from runtime arguments, so its topology, dimensions, coordinate
mapping, and routing directions are not normally compile-time constants. The
calculation occurs once per fabric operation, before arbitrary-length payload
packetization, rather than once per packet chunk. Its performance impact has
not yet been measured.

tt-lang does not use `experimental::FabricConnectionManager` because it
duplicates TT-Metal control-plane routing in each kernel and embeds a fixed
topology model. Host route resolution instead supports connection reuse and
future program-wide route optimization. The runtime-layout difference follows
from this decision.

tt-mlir's packet splitting, multicast, and semaphore helpers can still be
adapted to the direct manager without its topology model.

### EmitC and generated C++

`lib/Conversion/TTKernelToEmitC/TTKernelToEmitC.cpp` lowers routing-plane
operations to the direct TT-Metal API. The generated sender follows this
structure:

```cpp
tt::tt_fabric::RoutingPlaneConnectionManager connection_manager;
open_connections(connection_manager, connection_count, runtime_arg_base);

PacketHeaderPool::reset();
auto* packet_header = PacketHeaderPool::allocate_header(1);
fabric_set_unicast_route(connection_manager, packet_header, connection_slot);
#if !defined(FABRIC_2D)
packet_header->to_chip_unicast(static_cast<uint8_t>(chip_route));
#endif

packet_header->to_noc_fused_unicast_write_atomic_inc(...);
auto& sender = connection_manager.get(connection_slot).sender;
sender.wait_for_empty_write_slot();
sender.send_payload_without_header_non_blocking_from_address(...);
sender.send_payload_flush_blocking_from_address(...);

close_connections(connection_manager, connection_count);
```

The 1D call is conditional on TT-Metal's `FABRIC_2D` kernel define. The high-
level TTL program does not select this condition.

### Host runtime route binding

Host runtime route binding is the execution-setup code in
`python/ttl/kernel_runner.py`. It runs after compiled kernel specifications are
available and before the invocation calls `ttnn.generic_op(...)`. The binder
creates one `ProgramDescriptor` per logical device role and places those
descriptors into a `MeshProgramDescriptor`. For each generated kernel and
TENSIX node, it determines the active logical routes, resolves remote devices
with
`mesh_device.get_fabric_node_id()`, and calls
`ttnn.setup_routing_plane_connection(...)`.

The compiler-managed runtime prefix is:

```text
[
  connection_count,
  route_slot_0, ..., route_slot_N,
  chip_route_0, ..., chip_route_N,
  routing_plane_connection_manager_args...
]
```

`route_slot_I` selects the manager connection used by logical route `I`.
`chip_route_I` is the target packet-header value for that route. Connection
manager arguments begin at `1 + 2 * N`. Nodes without an active fabric route
receive zeroed route metadata and no connection-manager arguments.

This prefix is private to the tt-lang target runtime and generated TTKernel
code. It is not represented in TTL source attributes.

Compiler-managed global semaphore addresses follow the optional PipeNet SRAM
scratch base in common runtime arguments. They are allocated while constructing
the program descriptor. Route queries and synchronization allocation do not
execute in a device kernel or once per packet.

## Implementation details

### TT-Metal control-plane route query

`third-party/patches/ttmetal-expose-fabric-route-info.patch` adds a TTNN
binding for a control-plane route query. The query returns `FabricRouteInfo`,
which contains the selected forwarding-link index and physical hop count for a
source and destination `FabricNodeId` pair.

TT-Metal selects an available forwarding link and resolves the route through
`ControlPlane::get_fabric_route()`. The hop count is the number of physical
inter-node transitions in that resolved route. tt-lang passes the selected link
back to `setup_routing_plane_connection()` and passes the hop count to generated
TTKernel code as the packet-header route value.

This interface keeps topology interpretation in TT-Metal. tt-lang does not
infer routing from logical coordinates, physical mesh dimensions, node-number
differences, or architecture-specific topology tables.

Route resolution runs on the host while tt-lang constructs program
descriptors. Each `CompiledTTNNKernel` caches the resolved link and hop count
for a source-destination pair. The cache is cleared when the mesh object or
active fabric configuration changes. Connection semaphores and descriptor
runtime arguments remain per-invocation resources; no control-plane query runs
in a device kernel or per packet.

The tt-lang build applies managed patches from `third-party/patches` when
building the pinned TT-Metal source. Production toolchains include the same
TT-Metal change through the normal toolchain uplift process.

### Validated segmented transport behavior

The generated-C++ reference for each segment uses the fabric manager, packet
pool, fused write and atomic command, sender submission sequence, completion
wait, and connection closure.

The validated non-adjacent transfer does not calculate an arbitrary physical
hop count. It constructs a logical row-first and column-second sequence and
turns every adjacent pair into a separate transfer segment. Intermediate
devices receive and forward the payload. The P300 configuration uses
`to_chip_unicast(1)` for every segment.

tt-lang must obtain the segment sequence from its host-side route planner while
preserving the proven per-segment C++.

### Validation environment

Fabric tests use the standard tt-lang Docker environment and pytest invocation:

```bash
docker exec -w /home/bnorris/tt/tt-lang3 bnorris-ird-fabric \
  bash -lc 'source build-fabric/env/activate && \
  python -m pytest test/python/fabric -v'
```

Tests skip when fewer devices than their communication relation requires are
available. The collective suite derives its domain from the control-plane mesh
extent. The four-device container maps `/dev/tenstorrent/0` through
`/dev/tenstorrent/3` and currently reports a `2x2` extent.

### Current validation status

The following results have been observed on the four-device P300_X2 system:

| Test | Result | Evidence |
| --- | --- | --- |
| TT-Metal generic-op adjacent point-to-point | Pass | 4,096-byte BF16 payload delivered exactly. |
| Segmented reference `(0,0) -> (0,3)` | Pass | Three adjacent forwarding segments with fused payload and completion commands. |
| Standalone routing-plane unicast | Pass | Adjacent and two-router-hop transfers use host-resolved runtime hop counts. |
| Compiler point-to-point BF16 and FP32 | Pass | Logical `(0,0) -> (1,1)` executes twice on the discovered `2x2` domain. |
| Compiler ping-pong BF16 and FP32 | Pass | Forward and reverse transfers validated. |
| Compiler broadcast, scatter, gather, all-gather, and all-to-all | Pass | BF16 and FP32 results match exact host references on four devices. |
| Control-plane route integration unit tests | Pass | Kernel-runner tests validate selected-link and hop-count propagation. |
| Fabric completion storage lowering | Pass | Sender and receiver consume one runtime-bound global semaphore address; no local completion semaphore is emitted. |
| Full compiler fabric pytest suite | Pass | 14 passed in the four-device Docker container. |
| Full `test/python/pipe` regression suite | Pass | 77 passed and 1 expected failure. |

A source-level C++ match is not sufficient evidence of correctness. A fabric
feature is considered working only after its hardware pytest passes and the
existing pipe pytest suite shows no regression.

### Required remaining work

The POC still requires:

- a TT-Metal control-plane API that returns route segments or equivalent
  adjacency and routing data;
- compiler materialization of intermediate forwarding DFBs and synchronization
  for multi-segment routes;
- payload packetization using the runtime maximum payload size;
- receiver-published address support or a verified restriction to computed
  receiver DFB addresses;
- hardware pytests for reduce-to-root, reduce-scatter, and all-reduce in
  `test/python/fabric`;
- tree-reduction, packet-boundary, full-duplex, backpressure, and cache behavior
  hardware pytests;
- MLIR tests for TTKernel routing operations and generated C++;
- removal of temporary target queries after the corresponding TT-Metal APIs
  are available in the pinned revision.
