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
specific devices must be paired explicitly.

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

`Pipe.all_to_all(...)` records each selected source and destination pair, so it
supports rows, unions, and other subsets. When every device in a regular domain
participates and self-transfers are not required,
`TransferGraph.all_to_all(...)` represents the connections using the domain
extents and all-to-all parameters:

```python
from ttl.domains import TransferGraph

all_devices = TransferGraph.all_to_all(devices)
allgather = PipeNet(
    graph=all_devices,
    pipes=[Pipe(src=(0, 0), dst=(0, 0))],
)
```

This form avoids storing one source and destination coordinate pair for every
transfer.

## Transfer order and transport

Device-selected Pipes may contain both same-device and remote transfers. Their
callbacks process all same-device transfers first because those transfers use
NoC synchronization, while remote transfers use fabric synchronization.
Within each group, pairwise transfers follow the positions in their source and
destination views. All-to-all transfers process sources in parent-domain order
and, for each source, destinations in parent-domain order.

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
selected node.

The examples above establish frontend construction and compiler lowering.
Execution across two meshes and placement on devices with different Tensix
grids require separate device validation.
