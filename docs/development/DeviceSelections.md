# Device selections and Pipe endpoints

A device domain defines logical device coordinates. Indexing selects devices while retaining their coordinates in the parent domain. A node is a Tensix unit; node coordinates are separate from device coordinates.

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

`at_node(x, y)` creates a typed endpoint selection. It validates nonnegative node coordinates during construction. Operation placement must additionally check that the selected node exists on every selected device. Node selection does not impose a common grid on the device domain.

The existing local syntax `Pipe(src=(1, 0), dst=(0, 0))` retains its meaning. Complete point endpoints can instead specify devices directly:

```python
transfer = Pipe(
    src=cluster[0, 3, 2].at_node(11, 9),
    dst=cluster[1, 3, 2].at_node(0, 0),
)
```

Complete endpoints normalize to the existing graph PipeNet representation. Point pipes store one device edge; pairwise construction stores one edge per device pair and keeps the node specification separate. Equivalent parent endpoints have the same compilation identity, including when obtained through nested views.

The current graph verifier requires different source and destination devices. Complete endpoints on the same device, self-inclusive relations, and mixed local/fabric callback lowering require further compiler support. This restriction is in graph lowering, not domain membership. The examples here establish frontend construction; they do not establish two-mesh runtime or heterogeneous-grid execution support.
