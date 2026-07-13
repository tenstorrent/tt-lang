# Pipes on Fabric

This document describes the hardware capabilities, compiler design, software
architecture, implementation, and validation of cross-device PipeNets in
tt-lang. It complements [PipeNets](PipeNets.md), which defines the shared pipe
semantics used by both local NoC and fabric transports.

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
which device participates in a transfer. Target binding resolves that logical
device to a `FabricNodeId`. The packet's NoC command is built separately from
the destination node coordinates and receiver DFB address.

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
arrangement. The current P300_X2 system is described as a physical `2x2`
fabric mesh. Tests open those devices as a logical `1x4` `MeshDevice`. Logical
coordinate distance is therefore not a fabric hop count.

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
- Target binding resolves logical devices, physical routes, transport limits,
  and connection metadata.
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

### Target route resolution

Target binding converts one logical transfer edge into a resolved route. A
resolved route contains one or more transport segments. Each segment records:

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

tt-blaze's `cross_device_send` provides the working reference for segmented
forwarding. It decomposes a logical source-to-destination transfer into
adjacent logical devices. Each intermediate device receives into a local DFB,
waits for completion, then sends the payload to the next device. Every P300
linear-fabric packet uses `to_chip_unicast(1)`.

tt-lang should preserve those generated-kernel properties while replacing
Blaze's topology-specific row-first and column-second route construction with
a target control-plane route. The compiler representation remains more
general than Blaze:

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

The cross-device pipe compilation flow is:

```text
Python DeviceDomain, DeviceRef, TransferGraph, and PipeNet
  -> TTL device-domain and transfer attributes
  -> Pipe Transfer IR and shared PipeNet resource planning
  -> logical fabric-route records attached to generated kernels
  -> target runtime binding to MeshDevice and FabricNodeId values
  -> target-resolved connection slots and packet routes
  -> TTKernel routing-plane operations
  -> EmitC calls into tt::tt_fabric
  -> TTNN MeshProgramDescriptor execution
```

The first three stages are architecture-neutral. Fabric configuration and
physical routing enter only in target runtime binding and later emission.

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

TTKernel and EmitC modifications are maintained in commits whose subject
starts with `[ttkernel]`. This keeps the kernel-dialect portion independently
reviewable and cherry-pickable.

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

### Host runtime binding

`python/ttl/kernel_runner.py` creates one `ProgramDescriptor` per logical
device role and places those descriptors into a `MeshProgramDescriptor`.
For each generated kernel and TENSIX node, it determines the active logical
routes, resolves remote logical devices with
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

## Implementation details

### TT-Metal physical-mesh query patch

The pinned TTNN Python API exposes `MeshDevice.shape` and `FabricNodeId`, but
does not expose the physical fabric mesh dimensions maintained by TT-Metal.
Those dimensions are required to distinguish the logical `1x4` P300 test mesh
from its physical `2x2` fabric arrangement.

`third-party/patches/ttmetal-expose-physical-fabric-mesh-shapes.patch` adds a
read-only Python binding for the existing TT-Metal C++ query:

```cpp
tt::tt_fabric::get_physical_mesh_shapes()
```

The binding returns pairs containing a mesh identifier and physical dimensions
in row-major order. It changes no router, connection, or packet behavior.

The patch is preferable to parsing `TT_MESH_GRAPH_DESC_PATH` in Python.
TT-Metal already parses the descriptor, applies system discovery, and owns the
control-plane representation. Duplicating its protobuf parser in tt-lang would
create a second source of topology truth and would not account for runtime
control-plane decisions.

The tt-lang build already applies managed patches from `third-party/patches`
when building its pinned TT-Metal source. The patch does not modify an
installed tt-lang toolchain. Hardware experiments apply it only to the
isolated TT-Metal build under `/home/bnorris/tt/tt-blaze-baseline`.

The physical-dimension query is sufficient to validate and encode a direct
linear route that remains on one physical axis. It is not a general route
resolver. It cannot describe failed links, selected forwarding links, turns,
or multi-mesh routes. The intended final TT-Metal interface is a control-plane
query that returns a resolved sequence of fabric route segments. Once the
pinned TT-Metal revision exposes that interface, tt-lang should use it and
remove the narrower managed patch.

### Current linear-route resolver

The target runtime converts a `FabricNodeId::chip_id` to physical coordinates
using the physical mesh dimensions. For `FABRIC_1D`, it then:

1. Requires source and destination to belong to the same physical mesh.
2. Verifies both node identifiers are within that mesh.
3. Computes the coordinate distance on each physical axis.
4. Rejects a route whose endpoints differ on more than one axis.
5. Uses the distance on the single differing axis as `chipRoute`.

This resolver corrects two invalid approaches found by hardware testing:

- logical `MeshDevice` coordinate distance;
- absolute `FabricNodeId::chip_id` difference.

Both approaches can disagree with physical fabric distance.

The current resolver should not be extended with target-specific topology
tables in Python. New architectures and non-linear routes require richer
control-plane results from TT-Metal.

### tt-blaze reference behavior

tt-blaze is the generated-C++ reference for the fabric manager, packet pool,
fused write and atomic command, sender submission sequence, completion wait,
and connection closure.

Its `cross_device_send` implementation does not calculate arbitrary physical
hop counts. It constructs a logical row-first and column-second sequence and
turns every adjacent pair into a separate transfer segment. Intermediate
devices receive and forward the payload. The P300 adaptation adds
`to_chip_unicast(1)` to every segment because the original Blaze workload uses
a 2D torus configuration where `fabric_set_unicast_route()` supplies the
route.

tt-lang intentionally differs in route planning. Blaze's logical traversal is
specific to its known mesh and algorithm. tt-lang must obtain the segment
sequence from target binding while preserving the proven per-segment C++.

### Validation environment

Multi-device hardware tests run on the host through the shared-device
scheduler. The wrapper unsets `TT_VISIBLE_DEVICES`, waits for all devices to
be available, and prevents a test from accidentally opening only one card.

The compiler fabric pytest runner is:

```bash
/home/bnorris/soft/bin/tt-run-when-free \
  /home/bnorris/tt/tt-lang3/_examples/fabric/run_ttlang_fabric_pytest.sh \
  test/python/fabric/test_ccl.py -xvs
```

The isolated tt-blaze baseline runners are:

```bash
/home/bnorris/soft/bin/tt-run-when-free \
  /home/bnorris/tt/tt-blaze-baseline/run-generic-p2p.sh

/home/bnorris/soft/bin/tt-run-when-free \
  /home/bnorris/tt/tt-blaze-baseline/run-cross-device-send.sh
```

All runners use the P300_X2 four-device mesh descriptor, apply a timeout, and
write output to `/tmp/device_test.log`.

### Current validation status

The following results have been observed on the four-device P300_X2 system:

| Test | Result | Evidence |
| --- | --- | --- |
| TT-Metal generic-op adjacent point-to-point | Pass | 4,096-byte BF16 payload delivered exactly. |
| Adapted tt-blaze `(0,0) -> (0,3)` | Pass | Three adjacent forwarding segments with fused payload and completion commands. |
| Compiler adjacent point-to-point | Pass | Four devices opened; destination data validated. |
| Compiler adjacent ping-pong BF16 | Pass | Forward and reverse transfers validated. |
| Compiler adjacent ping-pong FP32 | Pass | Forward and reverse transfers validated. |
| Compiler direct non-adjacent transfer using logical distance | Timeout | Proves logical distance is not a valid route encoding. |
| Physical-mesh resolver unit tests | Pass | Included in 39 passing domain and kernel-runner tests. |
| Compiler non-adjacent transfer using physical route resolution | Pending | Requires isolated TTNN rebuild and hardware execution. |
| Full `test/python/pipe` regression suite | Pending | Required before implementation commits. |

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
- hardware pytests for all general collectives in `test/python/fabric`;
- full `test/python/pipe` regression validation in Docker;
- MLIR tests for TTKernel routing operations and generated C++;
- removal of temporary target queries after the corresponding TT-Metal APIs
  are available in the pinned revision.
