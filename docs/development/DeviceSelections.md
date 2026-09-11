# Device selections and Pipe endpoints

A device domain defines a logical coordinate space for devices. Its extents support ordinary multidimensional indexing. A device selection identifies any subset of that space, including nonrectangular sets, while retaining each member's parent coordinates. A node is a Tensix unit; node coordinates are separate from device coordinates.

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

Each mesh contains 32 logical devices. This declaration connects device `(0, row, column)`, node `(11, 9)`, to device `(1, row, column)`, node `(0, 0)`. It has 32 device pairs and one node-pipe specification. Execution still requires an operation and runtime placement covering the devices and nodes.

Integer indexing fixes an axis; slicing retains it. `second_mesh[3, 2]` and `cluster[1, 3, 2]` identify the same device. Views use Python range indexing semantics, including negative indices and strided or reversed slices. A view stores one integer or range per parent axis; indexing does not enumerate its devices.

Explicit selections and unions describe nonrectangular membership:

```python
selected = cluster.select([cluster[0, 0, 0], cluster[1, 7, 3]])
combined = selected | cluster[0, 0, :]
```

Explicit sets remove duplicates and iterate in parent coordinate order. They preserve device identity but do not define a correspondence to another set. Pairwise therefore requires coordinate-structured views with equal extents, rather than accepting arbitrary selections with equal member counts.

`at_node(x, y)` creates a typed endpoint selection. It validates nonnegative node coordinates during construction. A TT-Lang operation currently has one logical launch grid, so operation placement must additionally check that the selected node belongs to that grid. The logical grid does not encode the physical Tensix placement on each device.

The existing local syntax `Pipe(src=(1, 0), dst=(0, 0))` retains its meaning. Complete point endpoints can instead specify devices directly:

```python
transfer = Pipe(
    src=cluster[0, 3, 2].at_node(11, 9),
    dst=cluster[1, 3, 2].at_node(0, 0),
)
```

Complete endpoints normalize to the existing graph PipeNet representation. Point pipes store one device edge; pairwise construction stores one edge per device pair and keeps the node specification separate. Equivalent parent endpoints have the same compilation identity, including when obtained through nested views.

`all_to_all` connects every source selection member to every destination selection member. This per-row allgather uses four devices in each row of an 8-by-4 domain:

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

Each PipeNet has 16 transfers. Every device sends to four devices and receives from four devices. `include_self=True` adds four same-device transfers; the other 12 use fabric. The compiler represents those sets separately because same-device transfers lower to NoC operations and cross-device transfers lower to fabric operations. The callback and `destination_count()` semantics still describe all four transfers received by each device.

Fabric transfers cost more than on-device NoC transfers, but every remote allgather contribution must cross a device boundary. The current direct-edge lowering emits one fabric transfer for each ordered pair of distinct devices: `P * (P - 1)` transfers for `P` devices. A ring or tree implementation would require a separate collective schedule that forwards contributions between stages; `Pipe.all_to_all` does not currently construct that schedule.

If data must reach several nodes on each destination device, use one PipeNet for the device allgather into one node per device and another PipeNet for the on-device distribution. This avoids sending the same remote contribution over fabric once per destination node, but it does not reduce the interdevice transfer count above.

The examples establish frontend construction and compiler lowering. Two-mesh runtime execution and heterogeneous-grid placement require separate device validation.
