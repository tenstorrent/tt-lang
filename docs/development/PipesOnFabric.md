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
plane selects the injection connection, and device routing tables encode the
packet route from the final physical destination.

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
injection connection. Binding completes before
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

A `MeshDevice` logical arrangement is not necessarily the physical fabric
arrangement. General collective tests query the control-plane extent and
construct the logical `DeviceDomain` from it instead of encoding a fixed
extent. The extent defines logical membership and row-major ordering; it does
not define fabric adjacency.

The current host runtime uses an identity binding from flattened
`DeviceDomain` coordinates to the `MeshDevice` logical arrangement and
requires their extents to match. This is a target-binding restriction, not a
`DeviceDomain` invariant. A general binding maps each `DeviceRef` to a
`MeshCoordinate`, passes the original logical coordinates to device role
predicates, and uses the mapped coordinate with `get_fabric_node_id()` for
control-plane route resolution.

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
`StructuredTransfer`. Current derived forms include axis-neighbor, gather, and
scatter relations. Additional collectives should add semantic transfer forms
such as all-to-all without adding target topology fields.

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
- queries the outgoing direction and eligible forwarding links for each
  source-destination pair;
- reuses one injection connection for destinations with the same direction
  and a common forwarding link;
- assigns links by manager lifetime, allowing a proven receiver/sender
  ownership pair to reuse a link while requiring all other managers to use
  distinct links;
- supplies final destination identifiers for 2D routing or a validated hop
  count for 1D routing.

The resulting transport plan records the connection slot, target route, source
and destination TENSIX nodes, L1 addresses, payload constraints, and
synchronization objects. It contains no topology inferred from a `DeviceDomain`.

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

A graph PipeNet lowers to one ordered `PipeNetRecordsAttr` and one callback
region for each source or destination role. The callback receives a selected
record whose coordinate and logical-device fields come from immutable tables.
This keeps callback code independent of the global edge count; it does not
clone the callback for every transfer edge. The current representation still
stores one metadata row per edge and launch node combination and scans the
table on each logical device. Per-device record indices can reduce the scan
cost, while target-level structured descriptors can reduce metadata, without
changing the domain or callback semantics.

### Pipe lowering

`lib/Dialect/TTL/Transforms/PipeLowering.cpp` builds one fabric-route plan for
the module before lowering individual sends. The current POC records the
logical local device, logical remote device, and source node set for each
route. It assigns each fabric send a stable route index.

Separate send and receiver-post transport interfaces keep PipeNet protocol
planning independent of transport emission. The NoC implementations emit
same-device transfers and receiver address publication. The fabric
implementations emit receiver-readiness atomics, sender atomics, and fused
payload-write-plus-completion operations through the routing-plane manager.

The resource planner resolves each synchronization counter to an L1 address
before transport emission. The transport implementation consumes that address and
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

The send operations take a connection index, destination device id,
destination mesh id, and destination hop count. The connection index selects
the injection slot. The active target configuration selects the applicable
route encoding.

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
  [H] Query and cache the outgoing direction for each distinct
      source/destination pair.
  [H] Group destinations by direction, configure one injection connection per
      direction, and write the connection index and target route into runtime
      args.
  [D] Once per kernel invocation: open the configured connections.
  [D] Once per fabric operation: index the selected connection, encode the
      packet route for the active fabric mode, construct the packet command,
      and submit it.

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
| Route resolution | `[H]` Queries the outgoing direction and eligible links, reuses one connection per direction when its destinations have a common link, and assigns distinct links to concurrent connections. For 1D routing, it validates each logical hop and connects to the adjacent device. `[D]` Packet encoding uses the final destination in 2D mode or the validated hop count in 1D mode. | `[D]` Derives an outgoing direction and hop encoding from destination mesh and device identifiers for each fabric operation. |
| Transfer operands | `[C]` Routing-plane TTKernel operations represent a connection index, destination device id, destination mesh id, and destination hop count. `[D]` Generated kernels select the applicable route encoding for the active fabric mode. | `[C]` [Fabric TTKernel operations](https://github.com/tenstorrent/tt-mlir/blob/7a1e911f83ff5d380703309732d2a91e70104b07/include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td#L4227-L4320) represent destination mesh and device identifiers. `[D]` The wrapper resolves them. |
| Connection selection | `[D]` Generated C++ indexes the selected manager slot directly. | `[D]` The wrapper [searches active connection tags](https://github.com/tenstorrent/tt-mlir/blob/7a1e911f83ff5d380703309732d2a91e70104b07/include/ttmlir/Target/TTKernel/LLKs/experimental_fabric_api.h#L38-L76) for each fabric operation. |
| Runtime arguments | `[C]` The compiler assigns an explicit base for the connection descriptor block. `[H]` The binder appends connection slots, final destinations, and control-plane-produced connection arguments. `[D]` Connection setup reads them once per kernel invocation. | `[H]` The runtime serializes topology and connection descriptors into a [fixed fabric argument block](https://github.com/tenstorrent/tt-mlir/blob/7a1e911f83ff5d380703309732d2a91e70104b07/include/ttmlir/Target/TTKernel/LLKs/experimental_fabric_api.h#L105-L135). `[D]` Setup parses that block once per kernel invocation. |
| Topology model | `[C]` Logical TTL domains contain no fabric topology constants. `[H]` Host runtime binding queries TT-Metal for outgoing directions and connections. `[D]` Canonical TT-Metal routing tables determine packet forwarding. | `[H]` The runtime supplies a topology descriptor encoding [two dimensions, four directions, a 32-device limit, and line/ring/mesh/torus categories](https://github.com/tenstorrent/tt-mlir/blob/7a1e911f83ff5d380703309732d2a91e70104b07/include/ttmlir/Target/TTKernel/LLKs/experimental_fabric_topology_info.h#L14-L35). `[D]` Kernel helpers interpret it. |
| Current operation coverage | Provides the unicast atomic and fused write-plus-atomic operations required by fabric PipeNets. | Also provides arbitrary-length packetization and multicast write and semaphore helpers. |
| Per-operation kernel work | `[D]` Indexes a preselected connection and encodes either a 2D final destination or a 1D hop count. | `[D]` Performs [logical-position route calculation](https://github.com/tenstorrent/tt-mlir/blob/7a1e911f83ff5d380703309732d2a91e70104b07/include/ttmlir/Target/TTKernel/LLKs/experimental_fabric_1d_routing.h#L36-L108) and connection-tag lookup before constructing the packet command. |

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
| Architecture portability | `[C]` Logical domains and transfer graphs contain no topology categories or link counts. `[H]` The host runtime obtains connections from the active TT-Metal control plane and writes target routes into kernel runtime arguments. A new route encoding requires corresponding host binding and TTKernel/EmitC lowering support. | `[H]` The runtime serializes explicit line, ring, mesh, and torus models with fixed dimension and device limits. `[D]` The kernel wrapper interprets that model. Supporting a new topology requires changes to the topology descriptor, runtime serialization, and kernel routing helpers. |
| Extensibility | Logical transfers, route planning, connection reuse, and packet emission are separate components. New route scoring or packet operations do not change domain semantics, but the compiler/runtime argument contract must track TT-Metal API changes. | The wrapper gives TTKernel operations a compact destination-oriented API and already provides broader packetization and multicast helpers. Extending its topology model also increases on-device wrapper state and routing logic. |
| Kernel performance | `[H]` Direction and eligible-link queries or cache lookups occur before program submission. `[D]` Kernels directly index reused connection slots and encode one target route per operation. Current planning assigns valid links without scoring contention or workload traffic. | `[D]` Destination-oriented operations support runtime-selected destinations within the encoded topology. Each fabric operation reconstructs source and destination logical positions, performs topology-dependent hop calculation, and linearly searches active connection tags. This work is potentially material for small or frequent transfers. |
| Route optimization | `[H]` A late planner can use global program information when TT-Metal exposes multiple legal route or link candidates. | `[D]` For every fabric operation, kernel helpers calculate the direction and hop encoding from local topology and destination values. This limits access to program-wide traffic information. |
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

Host runtime route binding is the execution-setup code in
`python/ttl/kernel_runner.py`. It runs after compiled kernel specifications are
available and before the invocation calls `ttnn.generic_op(...)`. The binder
creates one `ProgramDescriptor` per logical device role and places those
descriptors into a `MeshProgramDescriptor`. For each generated kernel and
TENSIX node, it determines the active logical routes, resolves remote devices
with
`mesh_device.get_fabric_node_id()`, queries forwarding directions and eligible
links, and groups destinations by direction. It validates a distinct-link
assignment for all interfering managers before calling
`ttnn.setup_routing_plane_connection(...)` with explicit link indices.

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

tt-lang requires three TTNN bindings during host runtime route binding:

- `get_eth_forwarding_direction()` validates a source-destination pair and
  returns its outgoing direction;
- `get_forwarding_link_indices()` exposes TT-Metal's existing control-plane
  forwarding-link query to Python;
- `setup_routing_plane_connection()` validates an explicit link, allocates
  connection semaphores, adds kernel defines, and appends connection runtime
  arguments.

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
are modified.

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

The current implementation supports identity-mapped logical devices and
programs whose concurrent connection requests fit the available forwarding
links. It validates the complete target-binding plan before modifying program
descriptors and rejects unsupported resource schedules. General fabric support
still requires:

- explicit target binding from each logical `DeviceRef` to a
  `MeshCoordinate`;
- target-level router aggregation and connection reuse beyond the compiler's
  per-node manager intervals, including any transport-specific barriers;
- multicast lowering for graph transfers with device-range destinations;
- a receiver-address publication protocol for schedules that cannot prove
  computed receiver addresses.

### Remaining optimization and validation work

- Jointly score legal routes and links by hop count, availability, estimated
  contention, connection reuse, and barrier cost.
- Measure destination-table decoding, host connection setup, connection reuse,
  packetization, and worker placement against specialized communication
  kernels.
