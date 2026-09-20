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
  | Host runtime route binding                                |
  | FabricNodeIds, forwarding directions, injection links     |
  +-----------------------------------------------------------+
                         |
                         v
  +-----------------------------------------------------------+
  | Source TENSIX node                                        |
  | encode target route -> fabric write + atomic              |
  +-----------------------------------------------------------+
                         |
                         | TT-Metal fabric routing tables
                         v
  +-----------------------------------------------------------+
  | Destination device                                        |
  | completion wait -> destination DFB consumption            |
  +-----------------------------------------------------------+
```

High-level TTL records the logical transfer only. The active TT-Metal control
plane selects the injection connection, host binding supplies the target route
encoding, and device routing tables forward the packet.

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
device to a `FabricNodeId`, queries the outgoing direction, and asks the
control plane owned by the active `MeshDevice` runtime context to configure an
injection connection. When explicit link assignment is required, binding also
queries the eligible forwarding links. Binding completes before
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

Host runtime binding supplies the final physical destination for each logical
transfer. Generated kernels use the route encoding required by the active
fabric mode:

```cpp
#if defined(FABRIC_2D)
tt::tt_fabric::fabric_set_unicast_route(
    packet_header, destination_device_id, destination_mesh_id);
#else
tt::tt_fabric::fabric_set_unicast_route<false>(
    packet_header, destination_hop_count);
#endif
```

TT-Metal initializes routing tables when fabric starts. The 2D form encodes the
destination mesh and device. The 1D form encodes the number of hops along one
forwarding direction. The host validates each 1D hop against the control plane,
connects the manager to the first device, and supplies the final hop count. The
kernel does not reconstruct a topology model or search connection tags.

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
must not embed a device-specific limit. Larger PipeNet payloads require
target-aware packetization, with the completion increment in the final packet.

### Physical and logical device arrangements

A `DeviceDomain` defines a logical product index space. Component coordinates
are concatenated in component declaration order to produce a flattened tuple.
For example:

```python
devices = ttl.DeviceDomain.product(host=(2,), device=(4,))
```

maps `DeviceRef((host,), (device,))` to the tuple `(host, device)`. Component
names document logical roles; they do not select a host, chip, link, or fabric
axis.

For each materialized logical device, tt-lang currently performs this
logical-to-TTNN placement conversion:

1. Resolve the `DeviceRef` components in `DeviceDomain` declaration order.
2. Concatenate every component's axes into one integer tuple.
3. Construct a TTNN `MeshCoordinate` with that same tuple.

There is no coordinate permutation, offset, or placement lookup. The same
tuple selects the entry in the `MeshProgramDescriptor` and is passed to the
compiled device-role predicates. This document calls that conversion the
*coordinate-preserving placement*. It preserves the coordinate tuple only; it
does not establish physical-device identity or topology.

There is no rank adaptation. The flattened logical rank must equal the active
`MeshDevice` rank: tt-lang does not add, remove, combine, or split axes. When no
placement is specified, every logical coordinate is materialized and the
flattened domain extent must fit inside the active mesh. If the operation has
mesh tensors, their mesh extent must equal the flattened domain extent.
Explicit `mesh_program_placements` may select a subset inside the logical
domain, tensor mesh, and active mesh, but they do not remap coordinates. Every
PipeNet endpoint must remain covered. Incompatible ranks or selected
coordinates are rejected before program descriptors are constructed.

For the product domain above, the mapping stages are:

| Logical value | Flattened coordinate | TTNN placement | Physical identity |
| --- | --- | --- | --- |
| `DeviceRef(host=0, device=3)` | `(0, 3)` | `MeshCoordinate(0, 3)` | `mesh_device.get_fabric_node_id(MeshCoordinate(0, 3))` |
| `DeviceRef(host=1, device=0)` | `(1, 0)` | `MeshCoordinate(1, 0)` | `mesh_device.get_fabric_node_id(MeshCoordinate(1, 0))` |

tt-lang determines the first three columns through the coordinate-preserving
placement defined above. The active `MeshDevice` resolves the last column;
tt-lang does not derive a `mesh_id`, `chip_id`, adjacency, or distance from
`(host, device)`.

TTNN then maps each `MeshCoordinate` to a `FabricNodeId` with
`MeshDevice.get_fabric_node_id()`. This is where the logical mesh arrangement
first acquires a physical fabric identity. A `FabricNodeId::chip_id` is an
identifier within its physical mesh, not a distance or adjacency relation.

TTL currently has no user-facing placement contract that associates a logical
axis with a physical fabric axis or requires adjacent `DeviceDomain`
coordinates to map to adjacent `FabricNodeId` values. The
coordinate-preserving placement does not preserve adjacency. `axis_neighbor`
relations and stencil offsets therefore identify logical neighbors; they do
not guarantee a one-hop physical connection. `mesh_program_placements` only
restricts which logical coordinates execute and cannot remap them or impose an
adjacency constraint. Runtime route binding checks whether the selected fabric
mode can configure each resolved route, but that check is not a user-controlled
logical-to-physical placement guarantee.

A future language extension could support algorithms that intentionally
program the physical mesh. It would explicitly bind each logical `DeviceRef`
to a `MeshCoordinate` and express physical-axis or adjacency requirements for
the target to validate. That placement contract would remain separate from
the logical communication relation stored by `TransferGraph`. Direct physical
placement would make a program depend on a target topology and mesh extent,
reducing portability and automatic scaling. It should therefore be an optional
target-specific facility; topology-independent programs should continue to use
logical domains and let runtime binding select physical routes.

Route resolution depends on the active fabric mode:

- In 2D mode, host binding resolves only the source and final destination
  `FabricNodeId` values. TT-Metal's routing tables select all intermediate
  forwarding decisions.
- In 1D mode, the source and destination must differ along exactly one logical
  mesh axis. Host binding enumerates every intervening logical coordinate,
  maps each coordinate to a `FabricNodeId`, and requires every consecutive hop
  to use the same forwarding direction. Neighbor-exchange mode additionally
  requires exactly one logical hop. The generated packet encodes the validated
  hop count and opens its connection toward the first mapped hop.

For example, a 1D transfer from `(0, 0)` to `(0, 3)` enumerates `(0, 0)`,
`(0, 1)`, `(0, 2)`, and `(0, 3)`. The runtime maps all four coordinates to
physical node ids, validates that the three hops use one forwarding direction,
opens the connection toward the node for `(0, 1)`, and encodes hop count three.
A 1D transfer from `(0, 3)` to `(1, 0)` is rejected because two logical axes
differ. In 2D mode, that pair is legal when TT-Metal can route between the two
resolved `FabricNodeId` values; TT-Metal's routing tables select intermediate
routers.

The coordinate-preserving placement is an implementation restriction, not a
`DeviceDomain` invariant. Supporting arbitrary placement requires an explicit
mapping from each logical `DeviceRef` to a `MeshCoordinate`; inferring topology
from component names or coordinate differences would be incorrect.

## Design

### Design goals

Fabric pipes extend PipeNet communication across devices without making the
TTL programming model depend on a current Tenstorrent system topology. The
design has the following invariants:

- `DeviceDomain`, `DeviceRef`, and `TransferGraph` contain no
  fabric mode, physical mesh identifier, route direction, link index, packet
  limit, or NoC selection.
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

`DeviceRef` identifies one member of a device domain. `TransferGraph`
describes the logical communication relation. `PipeNet` applies the existing
pipe protocol to that relation.

An explicit `TransferGraph.edges(...)` relation enumerates arbitrary directed
logical-device pairs. It is appropriate for sparse or irregular connectivity
that has no structured constructor. Here, *explicit* refers only to logical
connectivity: it does not specify physical placement, links, or forwarding
routers. Regular axis-neighbor, stencil, gather, scatter, and all-to-all
relations use compact parameter-based forms so storage does not grow with the
number of derived edges.

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

Graph-only construction applies the transfer relation to every launch node. For
example, in an operation with a `(2, 2)` launch grid, the graph above describes
these four transfers:

```text
device (0, 0), node (0, 0) -> device (0, 3), node (0, 0)
device (0, 0), node (1, 0) -> device (0, 3), node (1, 0)
device (0, 0), node (0, 1) -> device (0, 3), node (0, 1)
device (0, 0), node (1, 1) -> device (0, 3), node (1, 1)
```

Each transfer uses the same node coordinate on its source and destination
device. A transfer between distinct node coordinates declares the node
relation separately:

```python
net = ttl.PipeNet(
    graph=transfers,
    pipes=[ttl.Pipe(src=(1, 0), dst=(0, 0))],
)
```

Each transfer is `(source device, source node) ->
(destination device, destination node)`. Device-indexed `ttl.Pipe` endpoints
allow one `PipeNet` to contain different device and node relations without a
separate public association type. PipeNet guards restrict which declared
endpoints execute protocol operations; they do not infer or modify the
relation.

The graph does not state whether the target uses a line, ring, torus, mesh, or
another interconnect. It also does not require `(0, 0)` and `(0, 3)` to be one
hardware packet apart.

`TransferGraph` supports explicit edge lists and parameter-based axis-neighbor,
stencil, gather, scatter, and all-to-all relations. Additional common relations
should describe communication semantics without adding target topology fields.

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

### Late route planning

The compiler does not infer or create physical routes from logical device
coordinates. Host runtime binding accepts only source and destination pairs
that the active TT-Metal control plane can route. The current implementation:

- resolves logical endpoints to `FabricNodeId` values;
- queries the outgoing direction along each resolved route and, when the
  target exposes link enumeration, the eligible injection links for its
  connection target;
- reuses one injection connection for destinations with the same direction
  and a common forwarding link;
- assigns links by manager lifetime, allowing a proven receiver/sender
  ownership pair to reuse a link while requiring all other managers to use
  distinct links;
- supplies final destination identifiers for 2D routing or a validated hop
  count for 1D routing.

Together, the compiler's fabric route and Pipe module plans record logical
route indices, source and destination TENSIX nodes, L1 address formulas,
payload constraints, and synchronization objects. Host binding maps each
logical route index to a connection slot and final physical target. Neither
plan infers topology from a `DeviceDomain`.

A future route optimizer belongs in this late planner. When the control plane
exposes multiple legal routes or links, the planner can reject candidates that
the selected packet format cannot encode, score the remaining candidates by
hop count, link availability, estimated contention, and connection reuse, and
cache the selected plan for the mesh and fabric configuration. A target that
cannot encode a legal route in one packet may materialize explicit forwarding
after logical transfer analysis; that is not required by the current
destination-routed TT-Metal transport.

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
  -> FabricNodeId and MeshDevice-scoped forwarding-direction queries
  -> control-plane connection setup and link selection
  -> per-device ProgramDescriptor runtime arguments
  -> MeshProgramDescriptor construction
  -> TTNN MeshProgramDescriptor execution
```

Logical transfer analysis remains independent of physical topology. Concrete
fabric configuration, connection selection, and final physical destinations
first enter during host runtime route binding, after kernel generation and
before `ttnn.generic_op(...)` submission. Generated kernels consume those
values as runtime arguments.

### Logical-to-physical binding timeline

The compiler preserves logical coordinates through Pipe lowering. Host
execution setup performs placement and physical binding:

```text
Compilation                              Host execution setup                         Device

DeviceRef -> DeviceRefAttr -> ttl.fabric_routes -> (1, 0) -> MeshCoordinate(1, 0)
 logical      logical          logical              logical    placement key
                                                              |
                                               active MeshDevice.get_fabric_node_id()
                                                              |
                                                              v
                              packet route <- route args <- FabricNodeId(mesh, chip)
                              target-specific             physical identity
```

For `DeviceRef(host=1, device=0)`, the stages are:

1. Frontend lowering creates a `DeviceRefAttr` containing component
   coordinates `((1,), (0,))`.
2. `PipeLowering.cpp` classifies the transfer as cross-device and attaches a
   `ttl.fabric_routes` entry to each affected kernel function. The entry stores
   logical `local` and `remote` `DeviceRefAttr` values, a stable route index,
   and source TENSIX nodes. It contains no physical identifier or route.
3. Artifact extraction concatenates the component coordinates in declaration
   order. The corresponding `FabricRouteSpec` contains the tuple `(1, 0)`.
4. `kernel_runner.py` creates the per-device descriptor at
   `MeshCoordinate(1, 0)`. It also passes `(1, 0)` separately as the logical
   coordinates consumed by compiled device-role predicates.
5. `fabric_target.py` calls
   `mesh_device.get_fabric_node_id(MeshCoordinate(1, 0))`. The returned
   `FabricNodeId(mesh_id, chip_id)` is the first physical device identity in
   this sequence; it is obtained from the active mesh, not calculated from the
   integers `1` and `0`.
6. Host binding queries forwarding information and fills the connection slot,
   destination device id, destination mesh id, and 1D hop-count tables. The
   generated kernel reads those tables to encode the packet route.

Keeping the two representations separate is intentional. Logical PipeNet IR
is independent of the machine that executes it, while topology, fabric mode,
and available links are properties of the active runtime context. The
separation also provides one explicit insertion point for a future compiler or
host placement planner: it can replace the current coordinate-preserving
conversion with a mapping from each logical `DeviceRef` to a
`MeshCoordinate`. Such a planner could preserve selected neighbor relations,
reduce hop count or contention, and account for unavailable devices without
changing the source-level transfer graph or TTL IR. Descriptor placement would
use the mapped physical coordinate, while device-role predicates would
continue to receive the original logical coordinate.

Embedding physical coordinates directly in portable source would bypass that
optimization point and make programs dependent on one mesh rank, extent, and
topology. A future direct-physical placement facility should therefore remain
explicit and target-specific.

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
  pipe;
- `TransferGraphAttr` for explicit edge lists or parameter-based
  logical-device relations;
- `PipeMappingAttr` for one device graph and its list of node Pipes; every
  graph edge is combined with every listed Pipe;
- `PipeNetRecordsAttr` for a local record list or an ordered list of graph
  mappings;
- `CurrentDeviceIndexOp` for the current member's row-major logical index.

These attributes contain no target route fields. Their verifiers check domain
membership, coordinate rank, and transfer structure.

A graph PipeNet lowers to one `PipeNetRecordsAttr` containing its mappings and
one callback region for each source or destination role. The callback receives
one selected transfer with node coordinates and logical device indices. Graphs
created with `axis_neighbor`, `stencil`, `gather`, `scatter`, or `all_to_all`
calculate endpoints from their stored parameters. An explicit graph compares
the current device index with the declared edge endpoints; its generated IR
scales with the number of declared edges rather than the complete domain.
Each device iterates only edges for which it is the source or destination.
Every transfer has a stable index used to select its resource-table entries.
Core specialization removes node-coordinate table columns whose value is
constant on that core. The frontend and TTL IR do not store a separate record
for every combination of device edge and node Pipe.

One compiled operation fixes its logical domain extents. A Python function may
accept domain extents and construct the corresponding graph for each supported
device count. Transfer graphs remain logical; host target binding resolves
physical placement, routes, and forwarding links.

### Pipe lowering

`lib/Dialect/TTL/Transforms/PipeLowering.cpp` first classifies every concrete
`DeviceTransferAttr`. A transfer whose source and destination devices are equal
uses the NoC transport. A cross-device transfer uses the fabric transport; its
sender and receiver-post sides each record the current logical device, remote
logical device, local injection TENSIX node, and owning function.

The resulting `ttl.fabric_routes` function attribute is deliberately logical.
Each dictionary entry contains `local` and `remote` `DeviceRefAttr` values,
`route_index`, and `source_nodes`. `PipeLowering` neither constructs a
`MeshCoordinate` nor queries topology. Python artifact extraction later
flattens the two device references into `FabricRouteSpec` tuples; physical
resolution still waits until host execution setup.

Before record-loop materialization, lowering builds immutable plans for every
graph callback. It then materializes those loops and expands high-level copies
into explicit transfer operations. From that stable transfer IR, it constructs
and validates a module-wide fabric plan before applying fabric metadata or
emitting TTKernel transport operations. Within each function it deduplicates
equal logical routes and assigns stable route indices.
Route indices address four aligned runtime tables: connection slot,
destination device id, destination mesh id, and 1D hop count. The selected
PipeNet record retains its concrete `PipeRecordAttr` during traversal, so later
queries reuse that record instead of expanding the complete transfer graph
again.

Manager lifetime analysis groups fabric operations into scoped intervals.
Lowering records which routes each interval uses, which generated receiver and
sender intervals may transfer ownership sequentially, and which intervals may
execute concurrently. It verifies matching locations, routes, and statically
bounded invocation counts before permitting ownership transfer. All remaining
interval pairs interfere. This interval plan is serialized as kernel metadata;
host binding uses it to assign forwarding links without mutating compiler IR.

After the complete plan is valid, lowering creates the routing-plane manager
operations and applies their route indices. Separate send and receiver-post
transport interfaces keep PipeNet protocol planning independent of transport
emission:

- Same-device receiver posts publish or compute a NoC destination and use
  local synchronization resources.
- Cross-device receiver posts send a reverse-route fabric atomic that publishes
  readiness to the sender.
- A cross-device sender waits for readiness, combines the host-provided
  destination DFB base with the destination TENSIX coordinates, and emits one
  fused fabric payload write plus completion increment on the forward route.
- The receiver waits on its local completion counter before consuming the DFB
  block.

The resource planner resolves every synchronization counter to an L1 address
before transport emission. Transport code consumes that address and never
interprets a PipeNet id as a semaphore id. Each cross-device transfer uses a
completion counter from the global semaphore namespace. Same-device completion
counters use dense local ids unless the selected allocation policy requires
global storage. If any sender-ready counter uses fabric, all ready-counter
colors use global storage so one color has one storage kind on every source
node. Proven receiver/sender manager ownership uses a separate local semaphore
and generation protocol.

Fabric lowering currently requires computed receiver DFB addresses. The
sender receives the destination DFB base from host runtime arguments and builds
the remote NoC address from the destination node coordinates. The
receiver-published address-table mechanism is limited to the NoC transport.
After packet injection, TT-Metal fabric routers perform all intermediate
forwarding; lowering does not generate programs for intermediate devices.

### TTKernel representation

TTKernel operations represent the routing-plane manager lifecycle and packet
submission:

- create a `RoutingPlaneConnectionManager` value;
- open connections from a runtime argument block;
- submit a remote atomic increment;
- submit a fused payload write and remote atomic increment;
- close the opened connections.

The send operations take a connection index, destination device id,
destination mesh id, and destination hop count. The connection index selects
the injection slot. The active target configuration selects the applicable
route encoding.

### EmitC and generated C++

`lib/Conversion/TTKernelToEmitC/TTKernelToEmitC.cpp` lowers routing-plane
operations to the direct TT-Metal API. The generated sender follows this
structure:

```cpp
tt::tt_fabric::RoutingPlaneConnectionManager connection_manager;
open_connections(connection_manager, connection_count, runtime_arg_base);

PacketHeaderPool::reset();
auto* packet_header = PacketHeaderPool::allocate_header(1);
#if defined(FABRIC_2D)
tt::tt_fabric::fabric_set_unicast_route(
    packet_header, destination_device_id, destination_mesh_id);
#else
tt::tt_fabric::fabric_set_unicast_route<false>(
    packet_header, destination_hop_count);
#endif

packet_header->to_noc_fused_unicast_write_atomic_inc(...);
auto& sender = connection_manager.get(connection_slot).sender;
sender.wait_for_empty_write_slot();
sender.send_payload_without_header_non_blocking_from_address(...);
sender.send_payload_flush_blocking_from_address(...);

close_connections(connection_manager, connection_count);
```

The destination encoder is selected by TT-Metal's `FABRIC_2D` kernel define.
The high-level TTL program does not select this condition.

### Host runtime route binding

`python/ttl/kernel_runner.py` orchestrates execution setup after compiled
kernel specifications are available and before the invocation calls
`ttnn.generic_op(...)`. It creates one `ProgramDescriptor` per materialized
logical device coordinate and places those descriptors into a
`MeshProgramDescriptor`.

`python/ttl/_src/fabric_target.py` owns target route resolution, complete-plan
validation, and descriptor mutation. For each generated kernel and TENSIX node,
it determines the active logical routes, maps remote coordinates with
`mesh_device.get_fabric_node_id()`, queries forwarding directions and eligible
links when the target exposes link enumeration, and groups destinations by
direction. It validates any required link assignment for interfering managers
before calling `ttnn.setup_routing_plane_connection(...)`. Noninterfering
managers may leave link selection to the control plane. No
semaphore, runtime argument, or program descriptor is modified until the
complete plan is valid.

An operation executes on its complete `device_domain` by default. The
`mesh_program_placements` operation option can instead select explicit logical
device coordinates or inclusive `ttl.MeshProgramPlacement` ranges. The host
then materializes descriptors only for those devices. Placement is explicit:
the compiler does not infer it from PipeNet endpoints. A device without a
PipeNet endpoint may still run a tensor copy or computation elsewhere in the
operation. Omitting that device would omit that work. Every graph-based PipeNet
source and destination must be included in the explicit placement. Explicit
placements must use one coordinate rank and their inclusive ranges must not
overlap.

The compiler-managed runtime prefix is:

```text
[
  connection_count,
  route_slot_0, ..., route_slot_N,
  destination_device_id_0, ..., destination_device_id_N,
  destination_mesh_id_0, ..., destination_mesh_id_N,
  destination_hop_count_0, ..., destination_hop_count_N,
  routing_plane_connection_manager_args...
]
```

`route_slot_I` selects the manager connection used by logical route `I`. The
remaining arrays describe its final target. Connection manager arguments begin
at `1 + 4 * N`. Nodes without an active fabric route receive zeroed route
metadata and no connection-manager arguments.

This prefix is private to the tt-lang target runtime and generated TTKernel
code. It is not represented in TTL source attributes.

Compiler-managed global semaphore addresses follow the optional PipeNet SRAM
scratch base in common runtime arguments. They are allocated while constructing
the program descriptor. Direction queries and synchronization allocation do not
execute in a device kernel or once per packet.

## Implementation details

### TT-Metal control-plane APIs

tt-lang uses these TTNN bindings during host runtime route binding:

- `get_eth_forwarding_direction()` validates a source-destination pair and
  returns its outgoing direction;
- `get_forwarding_link_indices()` exposes TT-Metal's existing control-plane
  forwarding-link query to Python when explicit link assignment is needed;
- `setup_routing_plane_connection()` validates an explicit link when supplied,
  otherwise selects the control-plane default, allocates connection semaphores,
  adds kernel defines, and appends connection runtime arguments.

Each `CompiledTTNNKernel` caches forwarding directions and eligible links by
source and destination `FabricNodeId`. The cache is cleared when the mesh
object or active fabric configuration changes. The binder first collects every
connection required by one source device. Within a manager, destinations with
the same direction reuse one connection only when their eligible-link sets
intersect. Across managers, the compiler records ownership intervals and an
interference graph. Deterministic graph coloring permits a proven
receiver/sender ownership pair to reuse a forwarding link and assigns distinct
links to all other managers. An external manager may reserve a fixed link
through operation runtime resources; tt-lang validates the reservation but does
not interpret or modify the external manager's runtime arguments. The complete
plan is validated before program descriptors, semaphores, or runtime arguments
are modified. If link enumeration is unavailable, noninterfering managers may
use the control-plane default; a plan that needs explicit link assignment is
rejected.

An external scoped manager call inside structured control flow records its
compiler-proven launch-node domain. Runtime binding resolves each kernel
descriptor independently, verifies that the recorded nodes are contained by
that descriptor, and unions the per-descriptor domains for the claim. An absent
domain means the complete descriptor executes the interval; a present empty
domain means no node executes it. The external binding must cover the resulting
node set exactly.

Each scoped external call releases its manager before returning. Scoped calls
in one RISC function are therefore sequential even when sibling structured
control-flow regions contain them, provided both execute on the same single
launch node. Multiple launch nodes may progress independently, and calls in
different functions may execute concurrently, so those intervals remain
interfering.

The compiler serializes an ordered sequence of generated receiver and sender
manager intervals only when corresponding intervals each contain one protocol
operation, implement the same transfers at the same device and TENSIX node,
use the same routes, and have equal statically bounded loop invocation counts.
PipeNet verification separately proves matching execution counts for every
send and receiver post. Unknown loop counts, mismatched interval sequences, or
generation-count overflow remain interfering and require distinct links.

One compiler-allocated local semaphore transfers ownership across the complete
proven sequence. The receiver opens its manager, publishes readiness, closes
the manager before waiting for payload completion, and publishes the sender
generation. The sender then opens its manager, sends the payload, closes the
manager, and publishes the next receiver generation. Repeated or
multi-interval sequences derive generations from a kernel-local invocation
ordinal that advances only when an interval executes. This preserves the
sequence when a conditional skips an interval. One single-shot interval uses
constant generations and requires no ordinal. Program submission reinitializes
compiler-managed semaphores, including when a cached program descriptor is
reused. Global-semaphore-only compilation does not allocate this local
ownership semaphore, so every manager remains interfering in that mode.

Connection setup still runs for each constructed program descriptor because
its semaphores and runtime arguments are invocation resources. Both the cached
direction query and connection setup run on the host before submission. No
control-plane query runs in a device kernel or once per packet.

Generated kernels pass final destination identifiers for 2D routing or a
validated hop count for 1D routing. This keeps forwarding decisions in TT-Metal
without adding a tt-lang-specific public route API or inferring physical routes
from node-number differences.

### Destination-routed transport behavior

Generated C++ uses the routing-plane manager, packet pool, target route encoder,
fused write and atomic command, sender submission sequence, completion wait,
and connection closure. Fabric routers forward the packet according to the
TT-Metal routing tables; intermediate TENSIX programs are not part of the
current transport.

### Validation requirements

Fabric changes require both a quick multi-device hardware check and a
representative full-system run. The full-system run uses the complete
discovered mesh and exercises the full fabric pytest suite. The smaller-system
run shortens the edit-test cycle but is not sufficient evidence of correctness.

The collective suite derives its domain from the control-plane mesh extent and
requests mesh routing so arbitrary physical turns remain one packet route.
Dedicated route tests request mesh, torus-X, torus-Y, and torus-XY modes. The
torus cases send across each configured boundary and require every tested
extent to exceed two, because a two-device extent cannot distinguish a wrap
link from its ordinary neighbor link. The bidirectional exchange test covers
that extent-two relation separately. A dedicated Galaxy CI fabric phase exposes
the complete discovered mesh so these cases exercise the system topology.
Target selection occurs when opening the runtime mesh, not in TTL domain or
transfer attributes.

Validation also includes the complete existing `test/python/pipe` suite and
the affected MLIR tests. A source-level C++ match or a smaller-system hardware
pass does not establish correctness without the full-system result.

### Remaining capability work

The current implementation supports coordinate-preserving logical-to-TTNN
placement and programs whose concurrent connection requests fit the available
forwarding links. It validates the complete target-binding plan before
modifying program descriptors and rejects unsupported resource schedules.
General fabric support still requires:

- a language-level physical-mesh placement facility that binds each logical
  `DeviceRef` to a `MeshCoordinate` and validates requested adjacency;
- target-level router aggregation and connection reuse beyond the compiler's
  per-node manager intervals, including any transport-specific barriers;
- tt-lang lowering and runtime binding for graph transfers with device-range
  destinations using TT-Metal fabric multicast;
- a receiver-address publication protocol for schedules that cannot prove
  computed receiver addresses.

### Remaining optimization and validation work

- Jointly score legal routes and links by hop count, availability, estimated
  contention, connection reuse, and barrier cost.
- Measure destination-table decoding, host connection setup, connection reuse,
  packetization, and node placement against specialized communication
  kernels.
