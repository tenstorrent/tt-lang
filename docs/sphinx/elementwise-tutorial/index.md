# Elementwise Kernel Tutorial

This tutorial walks through building a fused elementwise kernel in TT-Lang,
introducing one concept at a time. Each step is a self-contained runnable
script.

## The Goal

We want to compute `y = (a * b + c) * d` on 2048×2048 `bfloat16` tensors. The
inner expression `a * b + c` is the target for kernel fusion: instead of
dispatching three separate TT-NN operations that each read and write DRAM, a
custom TT-Lang kernel reads each input once, computes the result in L1, and writes
output once. It is possible to vary the expression as well as the size of
tensors and the data type, for example `float32`. We ecougarge the user to do this.

## Step 0 — TT-NN Baseline

**Script**: [`examples/elementwise-tutorial/step_0_ttnn_base.py`](https://github.com/tenstorrent/tt-lang/blob/main/examples/elementwise-tutorial/step_0_ttnn_base.py)

The starting point uses TT-NN directly, with no custom kernel:

```python
y = ttnn.multiply(ttnn.add(ttnn.multiply(a, b), c), d)
```

Each call dispatches a separate operation and writes an intermediate tensor back
to DRAM. This is the reference we'll verify against as we build the custom
kernel.

## Step 1 — Single Node, Single-Tile Block

**Script**: [`examples/elementwise-tutorial/step_1_single_node_single_tile_block.py`](https://github.com/tenstorrent/tt-lang/blob/main/examples/elementwise-tutorial/step_1_single_node_single_tile_block.py)

This step introduces the complete TT-Lang programming model. The kernel fuses
`a * b + c` into a single pass, processing one 32×32 tile at a time on one
node.

### Kernel function and grid

A kernel is a Python function decorated with `@ttl.kernel()`. The `grid`
argument selects how many nodes (Tensix cores) to run on. `grid=(1, 1)` means
a single node.

```python
@ttl.kernel(grid=(1, 1))
def __tutorial_kernel(a: ttnn.Tensor, b: ttnn.Tensor, c: ttnn.Tensor, y: ttnn.Tensor):
    ...
```

The function arguments are the tensors the kernel operates on. They live in
DRAM on device and are passed by the host at call time.

### Dataflow buffers

A *dataflow buffer* (DFB) is an L1 buffer shared between thread functions
within a node. It is created once in the kernel scope from a tensor likeness
and a block shape:

```python
a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
```

`shape=(1, 1)` means each buffer entry holds one 32×32 tile. `buffer_factor=2`
allocates two entries in L1 so that the reader and compute threads can work
concurrently — while compute processes one entry, the reader fills the other
(double-buffering).

### Thread functions

Three thread functions run concurrently inside the kernel:

```python
@ttl.compute()
def tutorial_compute(): ...

@ttl.datamovement()
def tutorial_read(): ...

@ttl.datamovement()
def tutorial_write(): ...
```

**Compute thread** — waits for filled input blocks and reserves output blocks,
then runs the fused expression:

```python
with (
    a_dfb.wait() as a_blk,
    b_dfb.wait() as b_blk,
    c_dfb.wait() as c_blk,
    y_dfb.reserve() as y_blk,
):
    y_blk.store(a_blk * b_blk + c_blk)
```

`wait()` blocks until the reader has pushed a filled tile. `reserve()` blocks
until the writer has freed an entry. The `with` block automatically calls `pop()`
on inputs and `push()` on the output when the scope exits.

**Reader DM thread** — copies tiles from DRAM into the input DFBs:

```python
with (
    a_dfb.reserve() as a_blk,
    b_dfb.reserve() as b_blk,
    c_dfb.reserve() as c_blk,
):
    tx_a = ttl.copy(a[row, col], a_blk)
    tx_b = ttl.copy(b[row, col], b_blk)
    tx_c = ttl.copy(c[row, col], c_blk)
    tx_a.wait(); tx_b.wait(); tx_c.wait()
```

`ttl.copy` starts a transfer; `tx.wait()` waits for it to complete. The
index `a[row, col]` selects a tile in *tile coordinates* (not element
coordinates). The `with` block calls `push()` on exit, signalling the compute
thread.

**Writer DM thread** — copies computed output tiles from L1 back to DRAM:

```python
with y_dfb.wait() as y_blk:
    tx = ttl.copy(y_blk, y[row, col])
    tx.wait()
```

## Step 2 — Single Node, Multi-Tile Block

**Script**: [`examples/elementwise-tutorial/step_2_single_node_multitile_block.py`](https://github.com/tenstorrent/tt-lang/blob/main/examples/elementwise-tutorial/step_2_single_node_multitile_block.py)

Processing one tile at a time incurs a synchronization (via dataflow buffers)
round-trip per tile. This step groups tiles into larger blocks so that each
transfer and compute iteration covers a `GRANULARITY × GRANULARITY` patch of tiles.

```python
GRANULARITY = 4  # each block is a 4×4 patch of 32×32 tiles = 128×128 elements

a_dfb = ttl.make_dataflow_buffer_like(
    a, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2
)
```

The iteration counts change from individual tiles to blocks:

```python
rows = a.shape[0] // TILE_SIZE // row_tiles_per_block
cols = a.shape[1] // TILE_SIZE // col_tiles_per_block
```

The reader selects a tile range (not a single tile) per transfer:

```python
tx_a = ttl.copy(
    a[start_row_tile:end_row_tile, start_col_tile:end_col_tile],
    a_blk,
)
```

The kernel structure, synchronization pattern, and compute expression are
unchanged from Step 1.

## Step 3 — Multi-Node, Fixed Grid

**Script**: [`examples/elementwise-tutorial/step_3_multinode.py`](https://github.com/tenstorrent/tt-lang/blob/main/examples/elementwise-tutorial/step_3_multinode.py)

This step parallelizes the kernel across a 4×4 grid of nodes. Each node
processes an independent rectangular region of the tensor. To familiarize
the user with Tenstorrent hardware architecture we recommend reading
[TT Architecture and Metalium Guide](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md).

### Declaring a multi-node grid

```python
@ttl.kernel(grid=(4, 4))
def __tutorial_kernel(...):
```

All nodes execute the same kernel body. They differentiate their work using
their coordinates in the grid as explained in the next sections.

### Querying grid size and node position

`ttl.grid_size(dims=2)` returns `(cols, rows)` — the number of nodes along
each grid dimension. `ttl.node(dims=2)` returns the `(col, row)` coordinates
of the current node, zero-based.

```python
grid_cols, grid_rows = ttl.grid_size(dims=2)

rows_per_node = a.shape[0] // TILE_SIZE // row_tiles_per_block // grid_rows
cols_per_node = a.shape[1] // TILE_SIZE // col_tiles_per_block // grid_cols
```

### Mapping local to global indices

Each DM thread uses its node coordinates to offset into the global tensor:

```python
node_col, node_row = ttl.node(dims=2)

for local_row in range(rows_per_node):
    row = node_row * rows_per_node + local_row
    ...
for local_col in range(cols_per_node):
    col = node_col * cols_per_node + local_col
    ...
```

In this particular example, the compute thread is unaware of node coordinates — it
simply processes all blocks that the DM threads deliver to it.

This version requires the tensor dimensions to be evenly divisible by the grid.
See Step 4 for a version that handles arbitrary sizes.

## Step 4 — Multi-Node, Auto Grid

**Script**: [`examples/elementwise-tutorial/step_4_multinode_grid_auto.py`](https://github.com/tenstorrent/tt-lang/blob/main/examples/elementwise-tutorial/step_4_multinode_grid_auto.py)

This step removes two constraints from Step 3: the hard-coded grid size and
the requirement for even divisibility.

### Auto grid

```python
@ttl.kernel(grid="auto")
```

`grid="auto"` lets the compiler select the largest grid that fits available
hardware resources. The kernel must work correctly for any grid the compiler may
choose as elaborated next.

### Ceiling division

When the number of blocks does not divide evenly across the grid, nodes at the
trailing edge would be left without work. Ceiling division ensures every block
is assigned to some node:

```python
rows_per_node = -(-rows // grid_rows)  # ceil(rows / grid_rows)
cols_per_node = -(-cols // grid_cols)  # ceil(cols / grid_cols)
```

### Bounds checking

Nodes at the trailing edge may be assigned more iterations than there are
actual blocks. All three thread functions guard per-block work:

```python
for local_row in range(rows_per_node):
    row = node_row * rows_per_node + local_row
    if row < rows:          # skip if past the end of the tensor
        for local_col in range(cols_per_node):
            col = node_col * cols_per_node + local_col
            if col < cols:  # skip if past the end of the tensor
                ...
```

The guard must appear in every thread function — compute, read, and write —
so that they all agree on exactly which blocks to process.
