# Device selections and Pipe endpoints

A multi-device Pipe endpoint identifies both a logical device and a Tensix
node on that device. `DeviceDomain` and its selections identify the devices;
`at_node(x, y)` adds the node coordinate.

## Connect matching positions in two device regions

Suppose two 8-by-4 device groups exchange data between devices at the same row
and column:

```python
from ttl.domains import DeviceDomain
from ttl.pipe import Pipe, PipeNet

cluster = DeviceDomain((2, 8, 4))
first_mesh = cluster[0, :, :]
second_mesh = cluster[1, :, :]

exchange = PipeNet([
    Pipe.pairwise(
        src=first_mesh.at_node(11, 9),
        dst=second_mesh.at_node(0, 0),
    ),
])
```

`cluster` contains 64 logical devices addressed as `(group, row, column)`.
`first_mesh` selects the 32 devices in group 0, and `second_mesh` selects the 32
devices in group 1. `Pipe.pairwise(...)` connects corresponding positions in
the two selections:

```text
device (0, row, column), node (11, 9)
    -> device (1, row, column), node (0, 0)
```

The integer `0` or `1` fixes the group coordinate and removes that axis from
subsequent indexing. The two slices preserve the row and column axes, so both
views have extents `(8, 4)`. Consequently, `second_mesh[3, 2]` identifies the
same device as `cluster[1, 3, 2]`.

Using a slice for the group axis produces a different view:

```python
one_group = cluster[1:2, :, :]
```

`one_group` has extents `(1, 8, 4)` because the slice keeps the group axis.
This distinction matters when indexing the view and when pairing it with
another view. `Pipe.pairwise(...)` requires equal extents so that every source
position has one unambiguous destination position.

Slices follow Python indexing rules, including negative indices and strided or
reversed ranges. The frontend stores one integer or range per domain axis; it
does not enumerate every selected device.

## Attach Tensix node coordinates

`at_node(x, y)` applies the same node coordinate to every selected device. It
checks that both coordinates are nonnegative. Operation placement later checks
that the node is inside the operation's launch grid.

A single remote transfer uses one selected device at each endpoint:

```python
transfer = Pipe(
    src=cluster[0, 3, 2].at_node(11, 9),
    dst=cluster[1, 3, 2].at_node(0, 0),
)
```

For communication within one device, the existing node-only syntax remains
valid:

```python
local_transfer = Pipe(src=(1, 0), dst=(0, 0))
```

## Select devices that do not form a rectangle

Slices describe rectangular device regions. Use `select(...)` when only
specific devices participate, and use `|` to combine selections:

```python
selected = cluster.select([cluster[0, 0, 0], cluster[1, 7, 3]])
combined = selected | cluster[0, 0, :]
```

These expressions return a `DeviceSet`. A `DeviceSet` removes duplicates and
iterates its devices in the parent `DeviceDomain`'s row-major coordinate order.
It identifies the participating devices but does not define source-to-destination
pairs. Pairing two such sets by their sorted positions could silently connect
unintended devices. `Pipe.pairwise(...)` therefore accepts rectangular views,
whose coordinates define the correspondence. Use individual point Pipes when
specific devices must be paired explicitly and each pair has its own node
endpoints. When several arbitrary device pairs share one node relation, use an
explicit `TransferGraph.edges(...)` relation and one node `Pipe` instead.

`Pipe.all_to_all(...)` accepts any device selection because it connects every
selected source to every selected destination; it does not require pairwise
correspondence.

## Connect every selected source and destination

The following example creates one four-device allgather for each row of an
8-by-4 domain:

```python
devices = DeviceDomain((8, 4))
row_allgathers = [
    PipeNet([
        Pipe.all_to_all(
            src=devices[row, :].at_node(0, 0),
            dst=devices[row, :].at_node(0, 0),
            include_self=True,
        ),
    ])
    for row in range(8)
]
```

Each PipeNet contains 16 transfers. Every device sends to four devices and
receives from four devices. `include_self=True` adds four same-device transfers;
the other 12 use fabric.

`Pipe.all_to_all(...)` supports rows, unions, and other subsets by recording
each selected source and destination pair. When both endpoints cover a complete
single-component domain and self-transfers are excluded, it automatically uses
the compact all-to-all relation instead. The same relation can be declared
directly when its node Pipes are supplied separately:

```python
from ttl.domains import TransferGraph

all_devices = TransferGraph.all_to_all(devices)
allgather = PipeNet(
    graph=all_devices,
    pipes=[Pipe(src=(0, 0), dst=(0, 0))],
)
```

Both forms avoid storing one source and destination coordinate pair for every
transfer in this complete-domain case.

## Transfer order and transport

Device-selected Pipes may contain both same-device and remote transfers. Each
`Pipe` constructor emits its same-device transfers before its remote transfers
because those groups use NoC and fabric synchronization, respectively. A
`PipeNet` preserves `Pipe` declaration order; it does not move all local
transfers ahead of remote transfers from earlier Pipes. Within each group,
pairwise transfers follow the positions in their source and destination views.
All-to-all transfers process sources in source-selection order and, for each
source, destinations in destination-selection order. A `DeviceView` follows
its axis ranges, with the last axis varying fastest. A `DeviceSet` is normalized
to the parent domain's row-major order: components and axes are flattened in
declaration order, with the last axis varying fastest. For example,
`DeviceDomain((2, 3))` orders devices as `(0, 0)`, `(0, 1)`, `(0, 2)`,
`(1, 0)`, `(1, 1)`, `(1, 2)`.

The direct all-to-all implementation emits one fabric transfer for each ordered
pair of distinct devices: `P * (P - 1)` transfers for `P` devices. A ring or
tree collective must instead declare the transfers for each forwarding stage.

When data must reach several nodes on each destination device, one PipeNet can
transfer each contribution to one node per device and a second PipeNet can
distribute it within the device. This avoids repeating the fabric transfer for
each destination node.

## Runtime requirements

`DeviceDomain` coordinates are logical; they do not select physical devices.
Runtime binding maps them to physical devices and fabric routes. The operation
must cover every selected logical device, and its launch grid must contain each
selected node. The current binding concatenates domain-component coordinates
in declaration order and uses that tuple unchanged as the TTNN mesh coordinate;
[Pipes on Fabric](PipesOnFabric.md) describes the complete physical mapping.

The examples above establish frontend construction and compiler lowering.
Execution across two meshes and placement on devices with different Tensix
grids require separate device validation.
