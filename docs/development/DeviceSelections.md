# Device selections and Pipe endpoints

A device domain defines logical coordinates for devices. Its extents support
ordinary multidimensional indexing. A device selection identifies members of
that domain, including nonrectangular sets, while retaining their original
coordinates. A node is a Tensix unit; node coordinates are separate from
device coordinates.

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

Each mesh contains 32 logical devices. This declaration connects device
`(0, row, column)`, node `(11, 9)`, to device `(1, row, column)`, node
`(0, 0)`. The same source and destination node coordinates apply to all 32
device pairs. Execution also requires an operation and runtime placement that
cover those devices and nodes.

Integer indexing fixes an axis; slicing retains it. `second_mesh[3, 2]` and
`cluster[1, 3, 2]` identify the same device. Views use Python range indexing
semantics, including negative indices and strided or reversed slices. A view
stores one integer or range per parent axis; indexing does not enumerate its
devices.

Explicit selections and unions describe nonrectangular membership:

```python
selected = cluster.select([cluster[0, 0, 0], cluster[1, 7, 3]])
combined = selected | cluster[0, 0, :]
```

Explicit sets remove duplicates and iterate in parent coordinate order. They
preserve device identity but do not define which member corresponds to a
member of another set. `Pipe.pairwise(...)` therefore requires rectangular
views with equal extents rather than arbitrary selections with equal member
counts.

`at_node(x, y)` associates one node coordinate with every selected device. It
validates nonnegative node coordinates during construction. A TT-Lang operation
currently has one logical launch grid, so operation placement also checks that
the selected node belongs to that grid. The logical grid does not encode the
physical Tensix placement on each device.

The existing local syntax `Pipe(src=(1, 0), dst=(0, 0))` retains its meaning.
A Pipe can instead specify device and node coordinates at both endpoints:

```python
transfer = Pipe(
    src=cluster[0, 3, 2].at_node(11, 9),
    dst=cluster[1, 3, 2].at_node(0, 0),
)
```

The compiler represents device-selected endpoints with graph PipeNet IR. A
point Pipe stores one device edge. `Pipe.pairwise(...)` stores each device pair
once and stores the shared source and destination node coordinates once.
Equivalent endpoints have the same compilation identity, including when they
are obtained through nested views.

When a pairwise or all-to-all relation contains both same-device and remote
transfers, callbacks process the same-device transfers first because they use
NoC rather than fabric synchronization. Within each set, pairwise transfers
retain pair order; all-to-all transfers use source order, then destination
order.

`Pipe.all_to_all(...)` connects every selected source device to every selected
destination device. This per-row allgather uses four devices in each row of an
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

Each PipeNet has 16 transfers. Every device sends to four devices and receives
from four devices. `include_self=True` adds four same-device transfers; the
other 12 use fabric.

`Pipe.all_to_all(...)` stores one explicit device edge per transfer. It is
appropriate for selected subsets such as one row. For an all-to-all relation
over an entire regular domain, `TransferGraph.all_to_all(...)` stores only the
domain and relation kind.

Fabric transfers cost more than on-device NoC transfers, but every remote
allgather contribution must cross a device boundary. The current direct-edge
lowering emits one fabric transfer for each ordered pair of distinct devices:
`P * (P - 1)` transfers for `P` devices. A ring or tree implementation requires
a separate collective schedule that forwards contributions between stages;
`Pipe.all_to_all(...)` does not construct that schedule.

If data must reach several nodes on each destination device, one PipeNet can
perform the device allgather into one node per device and another can perform
the on-device distribution. This avoids sending the same remote contribution
over fabric once per destination node, but it does not reduce the interdevice
transfer count above.

The examples establish frontend construction and compiler lowering. Two-mesh
runtime execution and heterogeneous-grid placement require separate device
validation.
