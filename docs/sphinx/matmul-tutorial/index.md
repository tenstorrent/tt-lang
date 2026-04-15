# Matmul Tutorial

This tutorial walks through building a fused matrix multiplication operation in
TT-Lang, introducing one concept at a time. Each step is a self-contained
runnable script.

## The Goal

We want to compute `y = relu(a @ b + c)` on 8192×8192 `bfloat16` tensors. The
entire expression — matrix multiply, bias add, and activation — is the target
for kernel fusion: instead of dispatching three separate TT-NN operations that
each read and write DRAM, a custom TT-Lang operation streams tiles from DRAM
into L1, accumulates the dot product across the K dimension, adds the bias, and
applies relu before writing the result back. Later steps scale this to multiple
nodes and multiple devices using data parallelism and K-sharding.

## Step 0 — TT-NN Baseline

**Script**: [`examples/matmul-tutorial/step_0_ttnn_base.py`](https://github.com/tenstorrent/tt-lang/blob/main/examples/matmul-tutorial/step_0_ttnn_base.py)

The starting point uses TT-NN directly, with no custom operation:

```python
y = ttnn.relu(ttnn.add(ttnn.matmul(a, b), c))
```

Each call dispatches a separate operation and writes an intermediate tensor back
to DRAM. This is the reference we'll verify against as we build the custom
operation. Correctness is measured with Pearson Correlation Coefficient (PCC)
rather than `allclose` because matmul accumulates bfloat16 rounding differently
from a reference float32 computation.

## Step 1 — Single Node, Single-Tile Block

**Script**: [`examples/matmul-tutorial/step_1_single_node_single_tile_block.py`](https://github.com/tenstorrent/tt-lang/blob/main/examples/matmul-tutorial/step_1_single_node_single_tile_block.py)

This step introduces the complete TT-Lang programming model. The operation fuses
`relu(a @ b + c)` into a single pass, processing one 32×32 tile at a time on
one node.

### Operation function and grid

An operation is a Python function decorated with `@ttl.operation()`. The `grid`
argument selects how many nodes (Tensix cores) to run on. `grid=(1, 1)` means
a single node.

```python
@ttl.operation(grid=(1, 1))
def __tutorial_operation(
    a: ttnn.Tensor, b: ttnn.Tensor, c: ttnn.Tensor, y: ttnn.Tensor
):
    ...
```

The function arguments are the tensors the operation operates on. They live in
DRAM on device and are passed by the host at call time.

### Dataflow buffers

A *dataflow buffer* (DFB) is an L1 buffer shared between kernel functions within
a node. It is created once in the operation scope from a tensor likeness and a
block shape:

```python
a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
```

`shape=(1, 1)` means each buffer entry holds one 32×32 tile. `block_count=2`
allocates two blocks in L1 so that the reader and compute kernels can work
concurrently — while compute processes one entry, the reader fills the other
(double-buffering).

Matmul needs one additional DFB that the elementwise tutorial does not use:
`acc_dfb` holds the running accumulator for the K-reduction. Because compute
both reads the previous partial sum and writes a new one in each k-step, two
slots in `acc_dfb` alternate in a ping-pong pattern:

```python
acc_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)
```

### Kernel functions

Three kernel functions run concurrently inside the operation:

```python
@ttl.compute()
def compute(): ...

@ttl.datamovement()
def read(): ...

@ttl.datamovement()
def write(): ...
```

**Reader DM kernel** — for each output tile `(m, n)`, first reads the bias
`c[m, n]` into `c_dfb`, then streams all k-tiles of `a` and `b` into their
DFBs:

```python
for m_tile in range(m_tiles):
    for n_tile in range(n_tiles):
        with c_dfb.reserve() as c_blk:
            ttl.copy(c[m_tile, n_tile], c_blk).wait()

        for k_tile in range(k_tiles):
            with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                tx_a = ttl.copy(a[m_tile, k_tile], a_blk)
                tx_b = ttl.copy(b[k_tile, n_tile], b_blk)
                tx_a.wait(); tx_b.wait()
```

`ttl.copy` starts a non-blocking transfer; `tx.wait()` waits for completion.
The index `a[m_tile, k_tile]` selects a tile in *tile coordinates* (not element
coordinates). The `with` block calls `push()` on exit, signalling the compute
kernel.

**Compute kernel** — initializes the accumulator to zero, accumulates
`a @ b` across all k-tiles, then adds the bias and applies relu:

```python
for _ in range(m_tiles):
    for _ in range(n_tiles):
        with acc_dfb.reserve() as acc_blk:
            acc_blk.store(ttl.math.fill(acc_blk, 0))  # zero the accumulator

        for _ in range(k_tiles):
            with (
                a_dfb.wait() as a_blk,
                b_dfb.wait() as b_blk,
                acc_dfb.wait() as pre_acc_blk,   # previous partial sum
            ):
                with acc_dfb.reserve() as acc_blk:
                    acc_blk.store(pre_acc_blk + a_blk @ b_blk)

        with c_dfb.wait() as c_blk, acc_dfb.wait() as acc_blk:
            with y_dfb.reserve() as y_blk:
                y_blk.store(ttl.math.relu(c_blk + acc_blk))
```

`ttl.math.fill(acc_blk, 0)` produces a block expression that fills a block
with a scalar value; `store()` materializes the expression. `wait()` blocks
until the reader has pushed a filled tile. `reserve()` blocks until the writer
has freed an entry. The `with` block automatically calls `pop()` on inputs and
`push()` on the output when the scope exits.

**Writer DM kernel** — copies completed output tiles from L1 back to DRAM:

```python
with y_dfb.wait() as y_blk:
    ttl.copy(y_blk, y[m_tile, n_tile]).wait()
```

## Step 2 — Single Node, Multi-Tile Block

**Script**: [`examples/matmul-tutorial/step_2_single_node_multitile_block.py`](https://github.com/tenstorrent/tt-lang/blob/main/examples/matmul-tutorial/step_2_single_node_multitile_block.py)

Processing one tile at a time incurs a synchronization round-trip per tile and
limits the hardware's ability to amortize compute setup overhead. This step
groups tiles into larger blocks so that each transfer and compute iteration
covers a multi-tile patch.

```python
M_GRANULARITY = 4
N_GRANULARITY = 4
K_GRANULARITY = 4
```

The DFB shapes must match the tile dimensions of each tensor operand, which
differ because the matmul operands have different roles:

```python
a_dfb = ttl.make_dataflow_buffer_like(
    a, shape=(m_tiles_per_block, k_tiles_per_block), block_count=2  # M×K
)
b_dfb = ttl.make_dataflow_buffer_like(
    b, shape=(k_tiles_per_block, n_tiles_per_block), block_count=2  # K×N
)
c_dfb = ttl.make_dataflow_buffer_like(
    c, shape=(m_tiles_per_block, n_tiles_per_block), block_count=2  # M×N
)
```

The iteration counts change from individual tiles to blocks, and the reader
selects a tile range per transfer:

```python
m_blocks = a.shape[0] // TILE_SIZE // m_tiles_per_block

tx_a = ttl.copy(
    a[start_m_tile:end_m_tile, start_k_tile:end_k_tile],
    a_blk,
)
```

The operation structure, synchronization pattern, and compute expression are
unchanged from Step 1.

## Step 3 — Multi-Node, Fixed Grid

**Script**: [`examples/matmul-tutorial/step_3_multinode.py`](https://github.com/tenstorrent/tt-lang/blob/main/examples/matmul-tutorial/step_3_multinode.py)

This step parallelizes the operation across a 4×4 grid of nodes. To familiarize
the user with Tenstorrent hardware architecture we recommend reading
[TT Architecture and Metalium Guide](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md).

### Declaring a multi-node grid

```python
@ttl.operation(grid=(4, 4))
def __tutorial_operation(...):
```

All nodes execute the same operation body. They differentiate their work using
their coordinates in the grid.

### Partitioning strategy

For matmul, the M×N output space is partitioned across the grid. The K
dimension is **not** partitioned: every node iterates over all k-blocks to
accumulate its own independent partial product. No inter-node communication is
required.

`ttl.grid_size(dims=2)` returns `(grid_n, grid_m)` — the number of nodes along
each dimension. `ttl.node(dims=2)` returns the `(node_n, node_m)` coordinates
of the current node, zero-based.

```python
grid_n, grid_m = ttl.grid_size(dims=2)

m_blocks_per_node = m_blocks // grid_m
n_blocks_per_node = n_blocks // grid_n
```

### Mapping local to global indices

Each DM kernel uses its node coordinates to offset into the global tensor:

```python
node_n, node_m = ttl.node(dims=2)

for local_m_block in range(m_blocks_per_node):
    m_block = node_m * m_blocks_per_node + local_m_block
    ...
for local_n_block in range(n_blocks_per_node):
    n_block = node_n * n_blocks_per_node + local_n_block
    ...
```

The compute kernel iterates over the same `m_blocks_per_node × n_blocks_per_node`
count as the DM kernels, but does not need to know the node's coordinates
directly — the DM kernels already stream only the relevant tiles into the DFBs.

This version requires the block counts to be evenly divisible by the grid.
See Step 4 for a version that handles arbitrary sizes.

## Step 4 — Multi-Node, Auto Grid

**Script**: [`examples/matmul-tutorial/step_4_multinode_grid_auto.py`](https://github.com/tenstorrent/tt-lang/blob/main/examples/matmul-tutorial/step_4_multinode_grid_auto.py)

This step removes two constraints from Step 3: the hard-coded grid size and
the requirement for even divisibility.

### Auto grid

```python
@ttl.operation(grid="auto")
```

`grid="auto"` lets the compiler select the largest grid that fits available
hardware resources. The operation must work correctly for any grid the compiler
may choose.

### Ceiling division

When the number of blocks does not divide evenly across the grid, nodes at the
trailing edge would be left without work. Ceiling division ensures every block
is assigned to some node:

```python
m_blocks_per_node = -(-m_blocks // grid_m)  # ceil(m_blocks / grid_m)
n_blocks_per_node = -(-n_blocks // grid_n)  # ceil(n_blocks / grid_n)
```

### Bounds checking

Nodes at the trailing edge may be assigned more iterations than there are
actual blocks. All three kernel functions guard per-block work:

```python
for local_m_block in range(m_blocks_per_node):
    m_block = node_m * m_blocks_per_node + local_m_block
    if m_block < m_blocks:          # skip if past the end of the tensor
        for local_n_block in range(n_blocks_per_node):
            n_block = node_n * n_blocks_per_node + local_n_block
            if n_block < n_blocks:  # skip if past the end of the tensor
                ...
```

The guard must appear in every kernel function — compute, read, and write —
so that they all agree on exactly which blocks to process.

## Step 5 — Multi-Device, Shard M

**Script**: [`examples/matmul-tutorial/step_5_multidevice_shard_m.py`](https://github.com/tenstorrent/tt-lang/blob/main/examples/matmul-tutorial/step_5_multidevice_shard_m.py)

This step scales the operation to multiple devices using SPMD
(Single-Program Multiple-Data) mode. The TT-Lang operation body is unchanged
from Step 4; only the tensor distribution across devices changes.

### Opening a mesh device

```python
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, n_devices))
```

A mesh device groups multiple physical devices into a single logical device.
`FabricConfig.FABRIC_1D` configures a 1D ring interconnect between devices.

### M-sharding strategy

The M dimension of the output is split evenly: device `i` computes rows
`i * (M/n_devices)` through `(i+1) * (M/n_devices) - 1`. Because each row of
the output only depends on the corresponding rows of `a` and `c`, and on the
full matrix `b`, no inter-device communication is needed.

```python
a = from_torch(a, ttnn.ShardTensorToMesh(mesh_device, dim=0))  # shard M rows
b = from_torch(b, ttnn.ReplicateTensorToMesh(mesh_device))     # replicate K×N
c = from_torch(c, ttnn.ShardTensorToMesh(mesh_device, dim=0))  # shard M rows
y = from_torch(y, ttnn.ShardTensorToMesh(mesh_device, dim=0))  # shard M rows
```

`ShardTensorToMesh(dim=0)` splits the tensor along its first dimension across
all devices. `ReplicateTensorToMesh` sends the same tensor to every device.

### Gathering results

After the operation, the per-device output shards are concatenated on the host:

```python
y = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
```

The TT-Lang operation runs identically on each device in SPMD mode — `grid="auto"`
applies independently per device, filling the full per-device grid.

## Step 6 — Multi-Device, Shard K

**Script**: [`examples/matmul-tutorial/step_6_multidevice_shard_k.py`](https://github.com/tenstorrent/tt-lang/blob/main/examples/matmul-tutorial/step_6_multidevice_shard_k.py)

This step changes the sharding strategy: instead of splitting M across devices,
the K (reduction) dimension is split. This allows the matrix multiply to be
parallelized along the contraction axis at the cost of requiring a reduction
step to combine results.

### K-sharding strategy

Each device computes a partial dot product over its K slice:

```
device i: y_i = a[:, K_i] @ b[K_i, :] + c_i
```

where `K_i` is the slice of K assigned to device `i`. The full result is
`y = sum(y_i)`.

```python
a = from_torch(a, ttnn.ShardTensorToMesh(mesh_device, dim=1))  # shard K cols
b = from_torch(b, ttnn.ShardTensorToMesh(mesh_device, dim=0))  # shard K rows
```

### Handling the bias

The bias `c` must only be added once, not once per device. To handle this
within the uniform SPMD model, a stacked tensor is constructed where device 0
receives the real `c` and all other devices receive zeros:

```python
replicated_cs = torch.zeros((M * n_devices, N), dtype=torch.bfloat16)
replicated_cs[:M, :] = c  # only the first M rows carry the real bias
replicated_cs = from_torch(replicated_cs, ttnn.ShardTensorToMesh(mesh_device, dim=0))
```

After sharding along `dim=0`, device 0 gets `c` and devices 1..n−1 get zeros,
so the summation `sum(a_i @ b_i + c_i)` correctly produces `a @ b + c`.

### Host-side reduction

Because the kernel produces partial sums, relu cannot be applied on-device.
The host collects the partial outputs and reduces them manually before
activating:

```python
partial_ys = ttnn.to_torch(partial_ys, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

y = torch.zeros((M, N), dtype=torch.bfloat16)
for i in range(n_devices):
    y += partial_ys[i * M : (i + 1) * M, :]

y = torch.relu(y)
```

The TT-Lang operation body drops the `ttl.math.relu` from Step 4 and stores
the raw `c_blk + acc_blk` result, deferring activation to after the reduction.

## Step 7 — Multi-Device, Shard K with All-Reduce

**Script**: [`examples/matmul-tutorial/step_7_multidevice_shard_k_all_reduce.py`](https://github.com/tenstorrent/tt-lang/blob/main/examples/matmul-tutorial/step_7_multidevice_shard_k_all_reduce.py)

This step replaces the host-side manual reduction from Step 6 with an
on-device all-reduce, keeping the result on the mesh and enabling the
activation to be applied on-device as well.

### All-reduce

```python
replicated_ys = ttnn.all_reduce(partial_ys)
replicated_ys = ttnn.relu(replicated_ys)
```

`ttnn.all_reduce` sums `partial_ys` across all devices using the TT-Fabric
interconnect. Each device ends up with the fully reduced M×N result — the
output is replicated rather than sharded. `ttnn.relu` is then applied
on-device to all replicas in parallel.

### Verifying replicated results

Because all-reduce replicates the result, every device holds a correct copy of
the full output. The verification loop checks each device's copy independently:

```python
replicated_ys = ttnn.to_torch(
    replicated_ys, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
)
for i in range(n_devices):
    y = replicated_ys[i * M : (i + 1) * M, :]
    pcc = ...
    assert pcc > 0.99
```

Compared to Step 6, this approach avoids the host round-trip for reduction and
moves the relu entirely on-device. The TT-Lang operation body is identical to
Step 6.
