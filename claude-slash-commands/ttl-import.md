---
description: Import and translate a CUDA, Triton, PyTorch kernel, or TTNN program to TT-Lang DSL
argument-hint: <kernel-file-or-code>
---

## Prerequisites

All tools are installed at `~/.claude/commands/tools/`. Use this full path when invoking them.

Before doing anything else, run the smoke test to verify your remote setup:
```bash
~/.claude/commands/tools/smoke-test.sh
```
If the smoke test fails, STOP. Do NOT continue. Ask the user to fix their remote setup first.

If the smoke test fails due to remote.conf not existing, STOP, offer to help them create it from remote.conf.example. Then work on getting the smoke test passing before exploring or continuing.

## Tools Available

NOTE: flags on run-test.sh must come before file argument. You can use --help if unsure on how to use.

NOTE: run-test.sh will copy the file. You do not need to copy the test file each time.

```bash
~/.claude/commands/tools/run-test.sh /path/to/kernel.py       # Run in functional simulator (default)
~/.claude/commands/tools/run-test.sh --hw /path/to/kernel.py  # Run on real hardware (final validation)
~/.claude/commands/tools/copy-file.sh /path/to/file.py        # Copy a file to the remote
~/.claude/commands/tools/remote-run.sh <command>               # Run an arbitrary command on the remote
```

By default, run-test.sh uses the functional simulator (`ttlang-sim`). Use `--hw` for real hardware. **Iterate with the simulator first.** Only move to `--hw` for final validation or if the simulator has a bug that blocks your work.

**Reading remote logs (output is saved, not streamed):**
```bash
~/.claude/commands/tools/remote-run.sh cat /tmp/ttlang_test_output.log        # Full log
~/.claude/commands/tools/remote-run.sh tail -100 /tmp/ttlang_test_output.log  # Last 100 lines
~/.claude/commands/tools/remote-run.sh cat /tmp/ttlang_test_output.log | grep -i "error"  # Search log
~/.claude/commands/tools/remote-run.sh cat /tmp/ttlang_initial.mlir           # Initial MLIR
~/.claude/commands/tools/remote-run.sh cat /tmp/ttlang_final.mlir             # Final MLIR
```

**NOTE:** Grep with quoted patterns containing spaces does not work via `remote-run.sh` due to quoting through the SSH+docker chain. Always pipe through grep locally: `remote-run.sh cat /path/to/file | grep "pattern"`

## Task

Translate the provided kernel or TTNN program to a TT-Lang DSL kernel. The primary goal is a working, correct kernel that can be tested and iterated on.

## Input

$ARGUMENTS

## Key Use Case: Fusing TTNN Operations

A common use case is taking a sequence of TTNN operations and fusing them into a single TT-Lang kernel for better performance. For example:

```python
# Original TTNN program (multiple ops, multiple round trips)
x = ttnn.exp(input)
y = ttnn.add(x, bias)
z = ttnn.relu(y)

# Fused TT-Lang kernel (single kernel, all ops in one compute function)
@ttl.kernel(grid=(1, 1))
def fused_kernel(input, bias, out):
    # ... setup CBs ...
    @ttl.compute()
    def compute():
        inp = input_dfb.wait()
        b = bias_dfb.wait()
        o = out_dfb.reserve()
        # All ops fuse into one compute body
        result = ttl.math.relu(ttl.math.exp(inp) + b)
        o.store(result)
        # ... pop/push ...
```

**When fusing TTNN ops:**
1. Identify the sequence of ops to fuse
2. Create one DFB per input tensor
3. Chain operations in a single compute function
4. TT-Lang will generate optimized fused code

## TT-Lang Programming Model

### Kernel Structure

Every TT-Lang kernel has exactly three threads that run concurrently:
1. **Compute thread** (`@ttl.compute()`): Math operations on tiles in L1
2. **Reader thread** (`@ttl.datamovement()`): Loads data from DRAM to dataflow buffers
3. **Writer thread** (`@ttl.datamovement()`): Writes data from dataflow buffers to DRAM

These threads synchronize via **dataflow buffers** (DFBs).

### Basic Kernel Template

```python
import ttl

@ttl.kernel(grid=(1, 1))
def add_kernel(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with lhs_dfb.wait() as l, rhs_dfb.wait() as r, out_dfb.reserve() as o:
            o.store(l + r)

    @ttl.datamovement()
    def dm_read():
        with lhs_dfb.reserve() as blk:
            tx = ttl.copy(lhs[0, 0], blk)
            tx.wait()
        with rhs_dfb.reserve() as blk:
            tx = ttl.copy(rhs[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()

# Call the kernel directly (no return ttl.Program)
# add_kernel(lhs_tensor, rhs_tensor, out_tensor)
```

### Using Context Managers (Preferred)

The `with` statement automatically handles `pop()` and `push()`:

```python
@ttl.compute()
def compute():
    with input1_dfb.wait() as a, input2_dfb.wait() as b:
        with output_dfb.reserve() as o:
            result = a + b
            o.store(result)
    # pop/push happens automatically at end of with block

@ttl.datamovement()
def dm_read():
    with input1_dfb.reserve() as blk:
        tx = ttl.copy(input1[0, 0], blk)
        tx.wait()
    # push happens automatically
```

### Dataflow Buffer API Reference

```python
# Create a dataflow buffer
dfb = ttl.make_dataflow_buffer_like(
    tensor,           # TTNN tensor to inherit dtype/layout from
    shape=(R, C),     # Block size in tiles (e.g., (2, 2) = 4 tiles per block)
    buffer_factor=2   # Factor of extra blocks in DFB (2 = double buffering) for pipelining
)

# Consumer operations (compute thread consumes data)
blk = dfb.wait()       # Block until data available, returns block
dfb.pop()              # Release block back to producer

# Producer operations (datamovement thread produces data)
blk = dfb.reserve()    # Block until space available, returns block
dfb.push()             # Signal data is ready for consumer

# Context manager (preferred - auto pop/push)
with dfb.wait() as blk:      # For consumers
    # use blk...
with dfb.reserve() as blk:   # For producers
    # fill blk...

# Block operations
blk.store(expr)             # Store result of expression into block
```

**DFB Shape = Block Size:** The `shape=(R, C)` parameter defines the **block size** in tiles. A block is the unit of data transferred between threads. For tensors larger than one block, use **loops** to iterate over multiple blocks:

Note: buffer factor is a pipeline hint, not a queue depth. Almost all kernels just use 2. You are able to push as many tiles into a CB as you want, it's just a datatype like array or queue, even a buffer_factor=1 dataflow buffer can support hundreds of tiles.

```python
# 128x128 tensor = 4x4 tiles, process in 2x2 blocks (4 iterations)
dfb = ttl.make_dataflow_buffer_like(tensor, shape=(2, 2), buffer_factor=2)

@ttl.datamovement()
def dm_read():
    for row in range(2):      # 2 row-blocks
        for col in range(2):  # 2 col-blocks
            with dfb.reserve() as blk:
                tx = ttl.copy(tensor[row*2:(row+1)*2, col*2:(col+1)*2], blk)
                tx.wait()

@ttl.compute()
def compute():
    for _ in range(4):  # Must match total iterations in dm_read
        with dfb.wait() as blk, out_dfb.reserve() as o:
            o.store(ttl.math.exp(blk))
```

## Available Operations

### Binary Operators

```python
result = a + b      # Element-wise addition
result = a - b      # Element-wise subtraction
result = a * b      # Element-wise multiplication
result = a / b      # Element-wise division
result = a @ b      # Matrix multiplication (equivalent to ttl.math.matmul(a, b))
```

### Binary Functions

```python
result = ttl.math.max(a, b)  # Element-wise maximum
result = ttl.math.min(a, b)  # Element-wise minimum
```

### Unary Functions (ttl.math.*)

```python
result = ttl.math.exp(x)      # Exponential
result = ttl.math.log(x)      # Natural logarithm
result = ttl.math.sqrt(x)     # Square root
result = ttl.math.rsqrt(x)    # Reciprocal square root (1/sqrt(x))
result = ttl.math.recip(x)    # Reciprocal (1/x)
result = ttl.math.tanh(x)     # Hyperbolic tangent
result = ttl.math.sigmoid(x)  # Sigmoid (1/(1+exp(-x)))
result = ttl.math.relu(x)     # ReLU (max(0, x))
result = ttl.math.abs(x)      # Absolute value
result = ttl.math.neg(x)      # Negation (-x)
result = ttl.math.floor(x)    # Floor
result = ttl.math.ceil(x)     # Ceil
result = ttl.math.sign(x)     # Sign (-1, 0, or 1)
result = ttl.math.selu(x, scale, alpha)  # SELU activation
result = ttl.math.fill(x, value)         # Fill block with scalar value (value must be a constant!)
```

### Matrix Multiplication

```python
# Two equivalent ways to do matmul:
result = a @ b                    # @ operator
result = ttl.math.matmul(a, b)   # function call

# Example usage:
with a_dfb.wait() as a_tile, b_dfb.wait() as b_tile, c_dfb.reserve() as c_out:
    c_out.store(a_tile @ b_tile)
```

**Multi-tile matmul:** When CBs hold multiple tiles (e.g., shape=(2, 2)), the compiler generates loops over K dimension and accumulates automatically. The DST register persists across K iterations, enabling proper accumulation. For example, with A[1,2] @ B[2,1] = C[1,1], the K=2 tiles accumulate correctly.

### Power (scalar integer exponent)

```python
# Raises each element to an integer power (top-level, not ttl.math)
result = ttl.power(x, 2)  # x^2
result = ttl.power(x, 3)  # x^3
```

### Transpose

```python
# Transpose tiles (top-level, not ttl.math)
# Takes input block, works with multi-tile CBs
with inp_dfb.wait() as x, out_dfb.reserve() as o:
    o.store(ttl.transpose(x))
```

**Non-square example:** For 4x2 tiles → 2x4 tiles:
```python
inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(4, 2), buffer_factor=2)
out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 4), buffer_factor=2)  # Swapped!
```

### Reductions (require scaler tensor)

```python
# Reductions are in ttl.math and need a "scaler" tensor (1x1 DFB of all 1.0s)
# dims=[0] = collapse rows, dims=[1] = collapse columns, dims=[0, 1] = scalar

# Scaler: 32x32 tile of 1.0s in a 1x1 DFB
scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)

with inp_dfb.wait() as i, scaler_dfb.wait() as s, out_dfb.reserve() as o:
    # Scalar reduction (sum/max entire DFB -> single value in output [0,0])
    o.store(ttl.math.reduce_sum(i, s, dims=[0, 1]))
    o.store(ttl.math.reduce_max(i, s, dims=[0, 1]))

    # Collapse rows (reduce along dim 0): (N, M) -> (1, M)
    o.store(ttl.math.reduce_sum(i, s, dims=[0]))

    # Collapse columns (reduce along dim 1): (N, M) -> (N, 1)
    o.store(ttl.math.reduce_sum(i, s, dims=[1]))
```

**Dimension semantics match PyTorch:**
- `dims=[0]` for reduce **collapses rows** (dim 0) - output shape [1, M]
- `dims=[1]` for reduce **collapses columns** (dim 1) - output shape [N, 1]

**Multi-tile reduce:** Reduces across ALL tiles in the input DFB. For example, a 4x1 tile input DFB reduced with `dims=[0, 1]` produces a single scalar value (in a 1x1 output DFB). The reduction sums all elements across all 4 tiles into position [0,0].

### Broadcast

```python
# Broadcast expands a smaller block to match a larger output shape
# dims=[0] = expand dim 0 (rows), dims=[1] = expand dim 1 (cols), dims=[0, 1] = broadcast scalar

with scalar_dfb.wait() as s, out_dfb.reserve() as o:
    # Broadcast 1x1 scalar to fill entire output block
    o.store(ttl.math.broadcast(s, dims=[0, 1]))

with row_dfb.wait() as r, out_dfb.reserve() as o:
    # Broadcast (1,M) row across N rows: dims=[0] expands dim 0
    o.store(ttl.math.broadcast(r, dims=[0]))

with col_dfb.wait() as c, out_dfb.reserve() as o:
    # Broadcast (N,1) column across M columns: dims=[1] expands dim 1
    o.store(ttl.math.broadcast(c, dims=[1]))
```

**Broadcast dimension semantics (match PyTorch):**
- `dims=[0]` for broadcast **expands dim 0** (copies row to all rows) - input (1, M) -> output (N, M)
- `dims=[1]` for broadcast **expands dim 1** (copies column to all columns) - input (N, 1) -> output (N, M)

Note: Reduce and broadcast use matching dims. `dims=[1]` reduce collapses columns to produce (N, 1), `dims=[1]` broadcast expands that column back to (N, M).

### Conditional Select (DO NOT USE - has simulator issues)

```python
# ttl.where exists but has known issues - avoid using it
# result = ttl.where(condition, true_val, false_val)  # BROKEN
```

### Operation Fusion

Operations chain automatically - no need for store/reload between ops:

```python
@ttl.compute()
def fused_compute():
    with input_dfb.wait() as a, bias_dfb.wait() as b, out_dfb.reserve() as o:
        # All these ops fuse into one efficient compute body
        x = ttl.math.exp(a)
        y = x + b
        z = ttl.math.sigmoid(y)
        result = ttl.math.relu(z)
        o.store(result)
```

**Limitation:** Ops that take DFB arguments (matmul, reduce, transpose, broadcast) cannot be fused with each other. Each must have its own `with` block and store. Broadcast cannot be fused with elementwise ops either.

**When fusion fails:** Use sequential `with` blocks to break the chain - you do NOT need separate kernels:

```python
@ttl.compute()
def compute():
    # CORRECT: Break into two with blocks (still one kernel!)
    with a_dfb.wait() as a, b_dfb.wait() as b, intermediate_dfb.reserve() as inter:
        inter.store(a @ b)

    with intermediate_dfb.wait() as inter, scaler_dfb.wait() as s, out_dfb.reserve() as o:
        o.store(ttl.math.reduce_sum(inter, s, dims=[0, 1]))
```

The compiler fuses 20+ elementwise ops in a single compute function without issues.

## Kernel Design: Minimize DRAM Traffic

**Strive for one fused kernel.** Multiple kernels are fine for incremental development and debugging, but each kernel boundary creates DRAM round-trips. For production:

- **One kernel = one DRAM read + one DRAM write** (ideal)
- **Two kernels = read → compute → write → read → compute → write** (2x DRAM traffic)
- **N kernels = N× DRAM traffic** (avoid)

```python
# BAD: Two kernels = 2x DRAM traffic
@ttl.kernel(grid=(1, 1))
def kernel1(inp, temp):
    # Read inp from DRAM, write temp to DRAM
    ...

@ttl.kernel(grid=(1, 1))
def kernel2(temp, out):
    # Read temp from DRAM, write out to DRAM
    ...

# GOOD: One fused kernel = 1x DRAM traffic
@ttl.kernel(grid=(1, 1))
def fused_kernel(inp, out):
    # Read inp from DRAM once, all compute in L1, write out to DRAM once
    # Use intermediate CBs (L1) instead of intermediate tensors (DRAM)
    ...
```

**Development workflow:** Start with multiple simple kernels to verify correctness, then fuse into one kernel for performance.

## Multi-Tile Processing and Streaming

**Strive to always use `grid="auto"` with streaming loops:**

- **`grid="auto"`** - this automatically selects the grid size at compile time. Hardcoded grids are only for special cases (e.g., pipe topologies that require a fixed core count). Using grid="auto" will enable full core utilization from the get go.
- **Stream with loops** in both compute and datamovement threads to handle arbitrary input sizes through DFBs.
- **Compute tiles_per_core dynamically** from tensor shape and grid size so kernels work on any input size.

Always strive to use the above patterns to ensure your kernels are flexible for any input size and fully utilize the cores available.

The exception: often for debugging or incremental development, it's helpful to start with a single core kernel; that is fine. You can start with a single core to isolate or debug a pattern, but strive to set it up in a way that it will naturally work with multiple cores later.

### IMPORTANT: Match the User's Target Data Size

**If the user provides a specific model config or tensor shape, strive to support that size.** You can simplify to smaller tensors for initial testing and debugging, but the goal is a kernel that works on their actual data. Use loops and streaming to handle large inputs:

```python
TILE_SIZE = 32
GRANULARITY = 4  # tiles per block dimension

@ttl.kernel(grid="auto")
def streaming_kernel(a, b, c, y):
    row_tiles_per_block = GRANULARITY
    col_tiles_per_block = GRANULARITY

    grid_cols, grid_rows = ttl.grid_size(dims=2)

    rows = a.shape[0] // TILE_SIZE // row_tiles_per_block
    cols = a.shape[1] // TILE_SIZE // col_tiles_per_block

    rows_per_core = -(-rows // grid_rows)  # divceil
    cols_per_core = -(-cols // grid_cols)  # divceil

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_col, core_row = ttl.core(dims=2)
        for local_row in range(rows_per_core):
            row = core_row * rows_per_core + local_row
            if row < rows:
                for local_col in range(cols_per_core):
                    col = core_col * cols_per_core + local_col
                    if col < cols:
                        with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk, c_dfb.wait() as c_blk, y_dfb.reserve() as y_blk:
                            y_blk.store(a_blk * b_blk + c_blk)

    @ttl.datamovement()
    def dm_read():
        core_col, core_row = ttl.core(dims=2)
        for local_row in range(rows_per_core):
            row = core_row * rows_per_core + local_row
            if row < rows:
                sr = row * row_tiles_per_block
                er = (row + 1) * row_tiles_per_block
                for local_col in range(cols_per_core):
                    col = core_col * cols_per_core + local_col
                    if col < cols:
                        sc = col * col_tiles_per_block
                        ec = (col + 1) * col_tiles_per_block
                        with a_dfb.reserve() as blk:
                            tx = ttl.copy(a[sr:er, sc:ec], blk); tx.wait()
                        with b_dfb.reserve() as blk:
                            tx = ttl.copy(b[sr:er, sc:ec], blk); tx.wait()
                        with c_dfb.reserve() as blk:
                            tx = ttl.copy(c[sr:er, sc:ec], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_col, core_row = ttl.core(dims=2)
        for local_row in range(rows_per_core):
            row = core_row * rows_per_core + local_row
            if row < rows:
                sr = row * row_tiles_per_block
                er = (row + 1) * row_tiles_per_block
                for local_col in range(cols_per_core):
                    col = core_col * cols_per_core + local_col
                    if col < cols:
                        sc = col * col_tiles_per_block
                        ec = (col + 1) * col_tiles_per_block
                        with y_dfb.wait() as blk:
                            tx = ttl.copy(blk, y[sr:er, sc:ec]); tx.wait()
```

From `examples/tutorial/multicore_grid_auto.py`. Key patterns: `grid="auto"`, dynamic `tiles_per_core` via divceil, bounds check with `if row < rows`.

**Key streaming principles:**
1. **DFB size is limited by L1** (~1.5MB per core) - you can't fit huge tensors
2. **Stream blocks through CBs** - read a block, process it, write it, repeat
3. **Loop counts must match** - compute iterations = dm_read iterations = dm_write iterations
4. **DRAM is large but slow** - keep data in L1 as long as possible, stream to avoid DRAM round-trips

## Pipes (Core-to-Core Communication)

Pipes are fully implemented in both the simulator and compiler. They enable core-to-core communication for patterns like gather, scatter, and ring exchanges. Get your kernel working without pipes first, then add them when needed for inter-core communication.

### Pipe API

```python
# Create pipes and wrap in a PipeNet
pipes = [ttl.Pipe((x, 0), ((x + 1) % N, 0)) for x in range(N)]
net = ttl.PipeNet(pipes)

# Send data through pipe (in dm_read on source core, inside a reserve block)
with dfb.reserve() as blk:
    tx = ttl.copy(src[0, 0], blk); tx.wait()
    def send(pipe):
        xf = ttl.copy(blk, pipe); xf.wait()
    net.if_src(send)

# Receive data from pipe (in dm_read on destination core)
with dfb.reserve() as blk:
    def recv(pipe):
        xf = ttl.copy(pipe, blk); xf.wait()
    net.if_dst(recv)
```

### Pipe Debugging Tips

- **Pipes cause hangs** when send/receive don't match - every `ttl.copy(blk, pipe)` needs a corresponding `ttl.copy(pipe, blk)`
- **Start without pipes** - get independent multi-core working first, then add pipes
- **Add pipes incrementally** - test after adding each pipe
- See the CB Threading Rules in the Debugging section for common deadlock causes

### Hardware Limits

- **32 CBs max** per core
- **~1.5MB L1** per core
- **~100MB total SRAM** across chip - utilize as much as possible for throughput
- **Tile size**: 32x32 elements = 2KB (bfloat16)

**Prefer `grid="auto"` with streaming** (shown above) over hardcoded grid sizes. See Reference Examples for complete working kernels.

## Tensor Setup

Tensors must be:
- **Tilized**: `layout=ttnn.TILE_LAYOUT` (32x32 element tiles)
- **Interleaved**: `ttnn.DRAM_MEMORY_CONFIG` or `ttnn.L1_MEMORY_CONFIG`
- **bfloat16**: Standard data type for Tenstorrent hardware

```python
import torch
import ttnn

device = ttnn.open_device(device_id=0)

# Create torch tensor (dimensions must be multiples of 32)
input_torch = torch.randn(64, 64, dtype=torch.bfloat16)
output_torch = torch.zeros(64, 64, dtype=torch.bfloat16)

# Convert to TTNN tensors
input_tensor = ttnn.from_torch(
    input_torch,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,  # or ttnn.L1_MEMORY_CONFIG
)
output_tensor = ttnn.from_torch(
    output_torch,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Run kernel
my_kernel(input_tensor, output_tensor)

# Read result back
result = ttnn.to_torch(output_tensor)

ttnn.close_device(device)
```

## Semantic Mapping: Think at the Hardware Level

**TT-Lang is a LOW-LEVEL DSL.** Do not expect a 1:1 mapping from PyTorch ops. When translating:

1. **Missing ops don't mean failure** - If `conv2d` doesn't exist, don't stop. Think about what conv2d *actually does* at the hardware level.

2. **Decompose to primitives** - Most "complex" operations are actually:
   - Simple compute (matmul, elementwise ops)
   - Complex data movement (gathering, reordering tiles)

3. **Data movement is the magic** - TT-Lang gives you full control over which tiles go where via `ttl.copy()` and tensor slicing. If you can describe WHERE data needs to go, you can implement the operation.

### Example: Conv2d

Conv2d seems like a "high-level op" but it's actually **matmul with clever data arrangement**:

```
What conv2d does:
- For each output position, gather a KxK window of input
- Flatten that window into a vector
- Dot product with filter weights

How to implement in TT-Lang:
- Reader kernel: Loop over output positions, DMA the KxK windows into CBs (im2col)
- Compute kernel: Just do matmul (window @ weights)
- Writer kernel: Write results back

The "conv2d" is in the data movement, not in a magic instruction.
```

### Example: Softmax

No `softmax` op? Decompose it: max → shift → exp → sum → divide

```python
# softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
# Numerically stable version with max subtraction

with x_dfb.wait() as x, scaler_dfb.wait() as s:
    # 1. Find max for numerical stability
    with max_dfb.reserve() as mx:
        mx.store(ttl.math.reduce_max(x, s, dims=[0, 1]))

    # 2. Broadcast max back to full size
    with max_dfb.wait() as mxv, bcast_dfb.reserve() as mxb:
        mxb.store(ttl.math.broadcast(mxv, dims=[0, 1]))

    # 3. Compute exp(x - max) and sum
    with bcast_dfb.wait() as max_bcast:
        shifted = x - max_bcast
        exp_shifted = ttl.math.exp(shifted)

        with sum_dfb.reserve() as sm:
            sm.store(ttl.math.reduce_sum(exp_shifted, s, dims=[0, 1]))

        # 4. Broadcast sum and divide
        with sum_dfb.wait() as sumv, sum_bcast_dfb.reserve() as smb:
            smb.store(ttl.math.broadcast(sumv, dims=[0, 1]))

        with sum_bcast_dfb.wait() as sum_bcast, out_dfb.reserve() as o:
            o.store(ttl.math.exp(x - max_bcast) / sum_bcast)
```

### Key Principle

When you are re-writing a high level operation or kernel:
1. **What does this kernel or op do at a HW level?** Think about what's actually happening in the HW when this op runs
2. **What primitives do we have?** matmul, elementwise, DMA with indexing
3. **Build it from primitives.** A naive O(n²) loop that works is better than giving up. The goal is NOT performance! Just correctness.
4. This is not a high level DSL like pytorch or ttnn, it's low level and you have explicit control over all of the HW, memory management, and synchronization. Do not think about direct mappings for high level ops and kernels, think about the best way to represent the kernel in tt-lang at the level it is designed to operate.

Even ops that DO exist may have different semantics (write in place, different numerical behavior). Always test to verify.

IMPORTANT: the test runner will just execute your script as a python file. Don't overthink it. The ttlang-sim and the hw runner will just run the script as python (not pytest!) so just **add a main block**, open device, print/assert tensor values. The sim should have full compatibility with ttnn function for moving tensors, opening device and so on:

Below will work on both hw and sim:
```
if __name__ == "__main__":
   device = ttnn.open_device(device_id=0)
   # call test functions here
   ttnn.close_device(device)
```

## Translation Guide: GPU → TT-Lang

### Concept Mapping

| GPU Concept | TT-Lang Equivalent |
|------------|-------------------|
| Thread block / workgroup | Grid of Tensix cores (`grid=(rows, cols)`) |
| Shared memory | L1 via dataflow buffers |
| Global memory | DRAM with DMA transfers |
| Warp/wave operations | Tile-level operations (32x32) |
| `__syncthreads()` | DFB `wait()`/`push()` synchronization |
| Kernel launch | Direct function call: `my_kernel(a, b, c)` |

### CUDA/Triton → TT-Lang

**Original CUDA pattern:**
```cuda
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

**TT-Lang equivalent:**
```python
@ttl.kernel(grid=(1, 1))  # Or multicore for large tensors
def add_kernel(a, b, c):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv:
            with c_dfb.reserve() as cv:
                result = av + bv  # Operates on entire 32x32 tile
                cv.store(result)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0, 0], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with c_dfb.wait() as blk:
            tx = ttl.copy(blk, c[0, 0])
            tx.wait()

# Call: add_kernel(a, b, c)
```

### PyTorch → TT-Lang

**Original PyTorch:**
```python
def gelu(x):
    return x * 0.5 * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x**3)))
```

**TT-Lang equivalent:**
```python
@ttl.kernel(grid=(1, 1))
def gelu_kernel(x, out):
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with x_dfb.wait() as xv:
            with out_dfb.reserve() as o:
                # Decompose GELU into available ops
                x3 = xv * xv * xv
                inner = xv + x3 * 0.044715  # Need scale tensor for constants
                # ... continue decomposition
                o.store(result)

    # ... dm_read, dm_write ...
```

**Note:** For scalar constants like 0.5, create a full tile tensor:
```python
scale_torch = torch.full((32, 32), 0.5, dtype=torch.bfloat16)
scale = ttnn.from_torch(scale_torch, ...)
```

## Using TTNN to Fill Gaps

If an operation isn't available in TT-Lang, you can use TTNN ops for:
- Input preprocessing (reshaping, padding, layout conversion)
- Operations not yet supported in TT-Lang
- Output post-processing

**Example: Using TTNN for padding**

```python
# TT-Lang requires tile-aligned dimensions (multiples of 32)
# Use TTNN to pad inputs that aren't tile-aligned

input_torch = torch.randn(100, 50)  # Not tile-aligned

# Pad to 128x64 (multiples of 32)
padded = ttnn.pad(input_tensor, padding=((0, 28), (0, 14)), value=0.0)

# Run TT-Lang kernel on padded input
my_kernel(padded, output_tensor)

# Slice result back to original size if needed
result = ttnn.slice(output_tensor, [0, 0], [100, 50])
```

**Rule of thumb:**
1. Try to implement in TT-Lang first
2. Use TTNN for preprocessing (padding, reshaping) and postprocessing (slicing)
3. The bulk of computation should be in TT-Lang for fusion benefits


## Iteration Workflow (REQUIRED)

**You MUST test every kernel you write.** The workflow has two phases:

### Phase 1: Iterate with the Functional Simulator (default)

The functional simulator (`ttlang-sim`) is the primary development tool. It catches DFB mismatches, shape errors, type errors, and functional bugs via dynamic analysis. Use it for all iteration.

```
1. Write kernel to file
2. Run: ~/.claude/commands/tools/run-test.sh /path/to/kernel.py
3. Read log: ~/.claude/commands/tools/remote-run.sh tail -100 /tmp/ttlang_test_output.log
4. If errors: fix and go to step 2
5. If success: verify numerical output is correct
```

### Phase 2: Validate on Real Hardware

Once the kernel passes in the simulator, do a final hardware run:
```
~/.claude/commands/tools/run-test.sh --hw /path/to/kernel.py
```

NOTE: it is possible that the sim and hw diverge which may require you to either use --hw early or iterate on a program that passes in the sim but not on HW. If your program works with the sim but not on HW you can use the same iteration flow from phase 1 to debug (you may need to isolate patterns and iterate). You can also ask the user for guidance, they may care more about HW or sim working.

**When to use `--hw` early:** If the simulator has a bug or is overly conservative for your use case, you can bypass it with `--hw` at any point. But prefer the simulator for iteration since it gives better error diagnostics.

**IMPORTANT:**
- Exit code 0 does NOT mean success - always read the log
- The log can be thousands of lines - use `tail`, `head` remotely, or pipe through `grep` locally (e.g., `remote-run.sh cat /tmp/ttlang_test_output.log | grep "pattern"`)
- Look for: `AssertionError`, `Exception`, `error:`, `FAIL`, `mismatch`
- Never guess at fixes - always read the actual error message
- **IMPORTANT:** Set a low timeout for faster iteration - tests should execute in under 1 second. Hangs are common (especially with pipes or DFB mismatches) and a low timeout helps detect them quickly.

**Handling Hangs:**
- If a kernel hangs, the most common cause is **DFB mismatch** - every `wait()` needs a corresponding `push()` from producer, every `reserve()` needs a corresponding `pop()` from consumer
- Verify loop counts match between compute and datamovement threads
- Kill zombie processes on remote: `~/.claude/commands/tools/remote-run.sh pkill -9 python`

## Compiler Errors: Workaround or Exit Early

**Your goal is NOT to debug the compiler.** If you hit an MLIR error or miscompile:

1. **First: Try a workaround**
   - Restructure the kernel differently
   - Use a different op combination
   - Split into multiple simpler kernels
   - Use TTNN for the problematic operation

2. **If no workaround exists: Exit early**
   - Report the error clearly to the user
   - Include the MLIR snippet that fails (from `/tmp/ttlang_initial.mlir` or `/tmp/ttlang_final.mlir`)
   - Describe what you tried
   - Do NOT spend time investigating compiler internals

**Signs of a compiler bug (not your fault):**
- MLIR verification errors
- Assertion failures in passes
- Segfaults during compilation
- Generated code that doesn't match the input semantics

## Low-Level DSL: Test Everything

**This is NOT PyTorch.** TT-Lang is a low-level DSL where you directly control memory management and synchronization. Operations may have unexpected semantics:

- Ops might write in place
- Ops might take dataflow buffers as arguments
- Ops might have different numerical behavior than PyTorch equivalents
- Memory layouts matter (tilized, interleaved, etc.)

**Do not assume PyTorch semantics.** If you're unsure how an op behaves, TEST IT.

### Debug Strategy: Isolate and Print

You cannot print or assert inside kernels. Instead:

1. **Test ops in isolation** - Write a minimal kernel with just one op
2. **Print tensors before/after** - Use `print(ttnn.to_torch(tensor))` after the kernel runs
3. **Compare against expected** - Compute the expected result in PyTorch and compare
4. **Build up incrementally** - Once one op works, add the next

```python
# Example: Testing an op in isolation
@ttl.kernel(grid=(1, 1))
def test_single_op(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as x:
            with out_dfb.reserve() as o:
                result = ttl.math.exp(x)  # Test just this one op
                o.store(result)
    # ... dm_read, dm_write ...

# After running:
print("Input:", ttnn.to_torch(inp_tensor))
print("Output:", ttnn.to_torch(out_tensor))
print("Expected:", torch.exp(inp_torch))
```

**Iterate as much as you need.** There is no limit on test runs. If behavior is unexpected, simplify further until you understand what's happening.

## Debugging Tips

1. **Start in isolation**: Test one op at a time before combining
2. **Print tensors**: Always print input/output to verify behavior
3. **Check shapes**: All dimensions must be multiples of 32
4. **Verify DFB balance**: Every `wait()` needs `pop()`, every `reserve()` needs `push()`
5. **Read the log**: Always check `/tmp/ttlang_test_output.log` after each run
6. **Check MLIR**: Use `/tmp/ttlang_initial.mlir` and `/tmp/ttlang_final.mlir` for compiler issues

### CB Threading Rules (Deadlock Debugging)

Each DFB has exactly one producer (`reserve`+push) and one consumer (`wait`+pop). The three threads (dm_read, compute, dm_write) all start simultaneously and run until they block.

- **Rule 1: One producer, one consumer per DFB.** A DFB flows between two threads (dm_read->compute or compute->dm_write) or is thread-local (compute->compute).
- **Rule 2: A DFB cannot have two producers.** If dm_read reserves on a DFB, compute CANNOT also reserve on it. Violation causes interleaved data or deadlock.
- **Rule 3: Thread-local accumulators must be initialized in compute, not DM.** The first iteration uses `reserve()` with an initial value; subsequent iterations use `wait()` + `reserve()` self-cycle.
- **Rule 4: Check every DFB appears in exactly two threads (or one if local).** For each DFB, list which threads call `reserve()` (producer) and `wait()` (consumer). If any DFB has `reserve()` in two different threads, that's a bug.

**If a kernel deadlocks**, check for DFBs that have `reserve()` in both dm_read and compute. That's the most common cause.

## Output

1. Save the translated TT-Lang kernel to a file
2. Run `~/.claude/commands/tools/run-test.sh` on the kernel and verify it passes in the simulator
3. Run `~/.claude/commands/tools/run-test.sh --hw` for final hardware validation
4. Read the log and confirm numerical correctness
5. Report any TTNN ops used to fill gaps
6. Only mark complete after the kernel runs successfully

---

## Reference Examples

### Example 1: Basic Pipe Send/Recv

Simplest pipe pattern. Core 0 loads a tile from DRAM and sends it to core 1 via pipe. Core 1 receives and writes to DRAM. Shows `PipeNet`, `if_src`, `if_dst` API. Note: pipes require a fixed grid size (not `grid="auto"`).

```python
@ttl.kernel(grid=(2, 1))
def pipe_send_recv(inp, out):
    net = ttl.PipeNet([ttl.Pipe((0, 0), (1, 0))])

    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        x, _ = ttl.core(dims=2)
        if x == 1:
            with inp_dfb.wait() as blk, out_dfb.reserve() as o:
                o.store(blk)

    @ttl.datamovement()
    def dm_read():
        x, _ = ttl.core(dims=2)
        if x == 0:
            with inp_dfb.reserve() as blk:
                tx = ttl.copy(inp[0, 0], blk); tx.wait()
                def send(pipe):
                    xf = ttl.copy(blk, pipe); xf.wait()
                net.if_src(send)
        if x == 1:
            with inp_dfb.reserve() as blk:
                def recv(pipe):
                    xf = ttl.copy(pipe, blk); xf.wait()
                net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        x, _ = ttl.core(dims=2)
        if x == 1:
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0, 0]); tx.wait()
```

From `test_pipe_basic.py`. Key points:
- Send happens inside `dm_read` after loading data, inside the same `reserve` block
- Receive happens in `dm_read` on the destination core
- Core 0 has no compute (it only sends); core 1 has no DRAM read (it only receives)

### Example 2: Ring Pipe (Neighbor Exchange)

Each core loads its own tile, sends it to the next core via a ring, receives its neighbor's tile, and adds them. This is the neighbor-sharing pattern used in molecular dynamics.

```python
N_CORES = 4

@ttl.kernel(grid=(N_CORES, 1))
def pipe_ring(inp, out):
    net = ttl.PipeNet([
        ttl.Pipe((x, 0), ((x + 1) % N_CORES, 0))
        for x in range(N_CORES)
    ])

    own_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    nbr_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with own_dfb.wait() as own, nbr_dfb.wait() as nbr, out_dfb.reserve() as o:
            o.store(own + nbr)

    @ttl.datamovement()
    def dm_read():
        x, _ = ttl.core(dims=2)
        with own_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, x], blk); tx.wait()
            def send(pipe):
                xf = ttl.copy(blk, pipe); xf.wait()
            net.if_src(send)
        with nbr_dfb.reserve() as blk:
            def recv(pipe):
                xf = ttl.copy(pipe, blk); xf.wait()
            net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        x, _ = ttl.core(dims=2)
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, x]); tx.wait()
```

From `test_pipe_ring.py`. Key points:
- Ring topology via `(x+1) % N_CORES` wraparound
- Every core both sends and receives (symmetric)
- `if_src`/`if_dst` dispatch to the correct pipe automatically per core

### Example 3: Scaled Dot-Product Attention (SDPA)

Single-core attention kernel showing the full softmax decomposition: Q@K^T, scale, row-wise max, shift, exp, sum, divide, then attn@V. Single-core is fine for getting the initial pattern working before scaling to multicore with streaming.

```python
SEQ_TILES = 1   # 32 tokens
HEAD_TILES = 2  # 64-dim head

@ttl.kernel(grid=(1, 1))
def sdpa_kernel(Q, K, V, scale, scaler, out):
    q_dfb = ttl.make_dataflow_buffer_like(Q, shape=(SEQ_TILES, HEAD_TILES), buffer_factor=1)
    k_dfb = ttl.make_dataflow_buffer_like(K, shape=(SEQ_TILES, HEAD_TILES), buffer_factor=1)
    v_dfb = ttl.make_dataflow_buffer_like(V, shape=(SEQ_TILES, HEAD_TILES), buffer_factor=1)
    scale_dfb = ttl.make_dataflow_buffer_like(scale, shape=(1, 1), buffer_factor=1)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)

    kt_dfb = ttl.make_dataflow_buffer_like(K, shape=(HEAD_TILES, SEQ_TILES), buffer_factor=2)
    qk_dfb = ttl.make_dataflow_buffer_like(Q, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    scaled_dfb = ttl.make_dataflow_buffer_like(Q, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    max_dfb = ttl.make_dataflow_buffer_like(Q, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    max_bcast_dfb = ttl.make_dataflow_buffer_like(Q, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    exp_dfb = ttl.make_dataflow_buffer_like(Q, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    sum_dfb = ttl.make_dataflow_buffer_like(Q, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    sum_bcast_dfb = ttl.make_dataflow_buffer_like(Q, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, HEAD_TILES), buffer_factor=1)

    @ttl.compute()
    def compute():
        # Step 1: K^T = transpose(K)
        with k_dfb.wait() as kv, kt_dfb.reserve() as kt:
            kt.store(ttl.math.transpose(kv))

        # Step 2: QK = Q @ K^T
        with q_dfb.wait() as qv, kt_dfb.wait() as ktv, qk_dfb.reserve() as qk:
            qk.store(qv @ ktv)

        # Step 3: QK_scaled = QK * scale
        with scale_dfb.wait() as s, qk_dfb.wait() as qkv:
            with scaled_dfb.reserve() as scd:
                bcast = ttl.math.broadcast(s, dims=[0, 1])
                scd.store(bcast * qkv)

        # Steps 4-6: Row-wise softmax
        with scaled_dfb.wait() as sdv, scaler_dfb.wait() as sc:
            with max_dfb.reserve() as mx:
                mx.store(ttl.math.reduce_max(sdv, sc, dims=[1]))
            with max_dfb.wait() as mxv, max_bcast_dfb.reserve() as mxb:
                mxb.store(ttl.math.broadcast(mxv, dims=[1]))
            with max_bcast_dfb.wait() as mxbv:
                with exp_dfb.reserve() as ex:
                    ex.store(ttl.math.exp(sdv - mxbv))
                with exp_dfb.wait() as exv, sum_dfb.reserve() as sm:
                    sm.store(ttl.math.reduce_sum(exv, sc, dims=[1]))
                with sum_dfb.wait() as smv, sum_bcast_dfb.reserve() as smb:
                    smb.store(ttl.math.broadcast(smv, dims=[1]))
                with sum_bcast_dfb.wait() as smbv, qk_dfb.reserve() as attn:
                    attn.store(ttl.math.exp(sdv - mxbv) / smbv)

        # Step 7: out = attn @ V
        with qk_dfb.wait() as av, v_dfb.wait() as vv, out_dfb.reserve() as o:
            o.store(av @ vv)

    @ttl.datamovement()
    def dm_read():
        with q_dfb.reserve() as blk:
            tx = ttl.copy(Q[0:SEQ_TILES, 0:HEAD_TILES], blk); tx.wait()
        with k_dfb.reserve() as blk:
            tx = ttl.copy(K[0:SEQ_TILES, 0:HEAD_TILES], blk); tx.wait()
        with v_dfb.reserve() as blk:
            tx = ttl.copy(V[0:SEQ_TILES, 0:HEAD_TILES], blk); tx.wait()
        with scale_dfb.reserve() as blk:
            tx = ttl.copy(scale[0, 0], blk); tx.wait()
        with scaler_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:HEAD_TILES]); tx.wait()
```

From `sdpa_kernel.py`. Key points:
- Row-wise softmax uses `dims=[1]` to collapse columns (per-row results), then `dims=[1]` broadcast to replicate back
- Transpose, matmul, reduce, broadcast each need their own `with` block and `store`
- `sdv` and `mxbv` are kept in scope via nesting so exp is recomputed rather than stored twice
- Single-core is fine for prototyping the compute pattern; add `grid="auto"` + streaming loops for production

### Example 4: Streaming Gating Kernel (grid="auto", RMSNorm, Reduce/Broadcast)

Real-world kernel from Engram model. Uses `grid="auto"` with `(1,1)` tile DFBs, streaming over sequence tiles. Shows RMSNorm via tile-by-tile reduce accumulation (looping over HIDDEN_TILES), dot product, gating, and the init-then-accumulate pattern for thread-local accumulators.

```python
TILE = 32
HIDDEN_TILES = 32  # 1024-dim / 32

@ttl.kernel(grid="auto")
def engram_gate_kernel(key, query, value, key_norm_w, query_norm_w,
                       scaler, mean_scale, inv_sqrt_d, eps_tile, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    seq_tiles = key.shape[0] // TILE
    tiles_per_core = -(-seq_tiles // grid_cols)

    key_dfb = ttl.make_dataflow_buffer_like(key, shape=(1, 1), buffer_factor=2)
    query_dfb = ttl.make_dataflow_buffer_like(query, shape=(1, 1), buffer_factor=2)
    value_dfb = ttl.make_dataflow_buffer_like(value, shape=(1, 1), buffer_factor=2)
    knw_dfb = ttl.make_dataflow_buffer_like(key_norm_w, shape=(1, 1), buffer_factor=2)
    qnw_dfb = ttl.make_dataflow_buffer_like(query_norm_w, shape=(1, 1), buffer_factor=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)
    isd_dfb = ttl.make_dataflow_buffer_like(inv_sqrt_d, shape=(1, 1), buffer_factor=1)
    eps_dfb = ttl.make_dataflow_buffer_like(eps_tile, shape=(1, 1), buffer_factor=1)

    sq_dfb = ttl.make_dataflow_buffer_like(key, shape=(1, 1), buffer_factor=2)
    red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    bcast_dfb = ttl.make_dataflow_buffer_like(key, shape=(1, 1), buffer_factor=2)
    dot_dfb = ttl.make_dataflow_buffer_like(key, shape=(1, 1), buffer_factor=2)
    gate_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        with scaler_dfb.wait() as sc, ms_dfb.wait() as ms, isd_dfb.wait() as isd, eps_dfb.wait() as eps:
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    # RMSNorm pass 1: sum of squares over HIDDEN_TILES
                    with key_dfb.wait() as k0:
                        with sq_dfb.reserve() as sq:
                            sq.store(k0 * k0)
                    with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                        r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                    with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                        acc.store(rv)
                    for j in range(HIDDEN_TILES - 1):
                        with key_dfb.wait() as kj:
                            with sq_dfb.reserve() as sq:
                                sq.store(kj * kj)
                        with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                            r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                        with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as new_acc:
                            new_acc.store(av + rv)

                    # Broadcast + rsqrt for normalization factor
                    with acc_dfb.wait() as total, bcast_dfb.reserve() as bc:
                        bc.store(ttl.math.broadcast(total, dims=[1]))
                    with bcast_dfb.wait() as bv, red_dfb.reserve() as scaled:
                        scaled.store(bv * ms)
                    with red_dfb.wait() as msq, red_dfb.reserve() as rsq:
                        rsq.store(ttl.math.rsqrt(msq))

                    # ... (query RMSNorm, dot product, gate, gate*value follow same pattern)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        # Load constants once (buffer_factor=1)
        with scaler_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()
        with ms_dfb.reserve() as blk:
            tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()
        with isd_dfb.reserve() as blk:
            tx = ttl.copy(inv_sqrt_d[0, 0], blk); tx.wait()
        with eps_dfb.reserve() as blk:
            tx = ttl.copy(eps_tile[0, 0], blk); tx.wait()
        # Stream tiles per core
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                for j in range(HIDDEN_TILES):
                    with key_dfb.reserve() as blk:
                        tx = ttl.copy(key[tile_idx, j], blk); tx.wait()
                # ... (query, key+weights interleaved, value tiles follow)

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                for j in range(HIDDEN_TILES):
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[tile_idx, j]); tx.wait()
```

From `engram_demo_ttlang.py`. Key patterns:
- **`grid="auto"` + streaming**: `tiles_per_core = -(-seq_tiles // grid_cols)` handles any sequence length
- **Init-then-accumulate**: First tile initializes `acc_dfb` via `reserve()`, remaining tiles do `wait()` + `reserve()` self-cycle (Rule 3)
- **`(1,1)` tile DFBs with loops**: Iterates over `HIDDEN_TILES` to reduce a full row, one tile at a time
- **Constants loaded once**: `buffer_factor=1` DFBs for scaler/eps/etc. loaded in dm_read, held in scope across all iterations in compute
- **dm_read must produce tiles in exact order compute consumes them**: key tiles for RMSNorm, then key+weights interleaved for dot product, etc.

### Example 5: Pipe Convolution (Forward Chain + Streaming)

Dilated 1D convolution using a forward pipe chain. Each core processes its sequence tiles and pipes boundary data to the next core. Shows pipes combined with streaming loops and the SiLU activation pattern.

```python
N_CONV_CORES = 4
HIDDEN_TILES = 32

@ttl.kernel(grid=(N_CONV_CORES, 1))
def pipe_conv_kernel(s0, s1, s2, s3, w0, w1, w2, w3, out):
    seq_tiles = s0.shape[0] // TILE
    tiles_per_core = -(-seq_tiles // N_CONV_CORES)

    pipes = [ttl.Pipe((x, 0), ((x + 1), 0)) for x in range(N_CONV_CORES - 1)]
    net = ttl.PipeNet(pipes)

    s0_dfb = ttl.make_dataflow_buffer_like(s0, shape=(1, HIDDEN_TILES), buffer_factor=2)
    s1_dfb = ttl.make_dataflow_buffer_like(s1, shape=(1, HIDDEN_TILES), buffer_factor=2)
    s2_dfb = ttl.make_dataflow_buffer_like(s2, shape=(1, HIDDEN_TILES), buffer_factor=2)
    s3_dfb = ttl.make_dataflow_buffer_like(s3, shape=(1, HIDDEN_TILES), buffer_factor=2)
    w0_dfb = ttl.make_dataflow_buffer_like(w0, shape=(1, HIDDEN_TILES), buffer_factor=1)
    w1_dfb = ttl.make_dataflow_buffer_like(w1, shape=(1, HIDDEN_TILES), buffer_factor=1)
    w2_dfb = ttl.make_dataflow_buffer_like(w2, shape=(1, HIDDEN_TILES), buffer_factor=1)
    w3_dfb = ttl.make_dataflow_buffer_like(w3, shape=(1, HIDDEN_TILES), buffer_factor=1)
    acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HIDDEN_TILES), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HIDDEN_TILES), buffer_factor=2)
    bnd_dfb = ttl.make_dataflow_buffer_like(s0, shape=(1, HIDDEN_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        # Non-zero cores receive boundary tile from previous core first
        if core_x > 0:
            with bnd_dfb.wait() as bnd, acc_dfb.reserve() as ctx:
                ctx.store(bnd)
            with acc_dfb.wait() as ctx, out_dfb.reserve() as o:
                o.store(ctx)
        # Weighted sum of 4 shifted inputs + SiLU activation
        with w0_dfb.wait() as cw0, w1_dfb.wait() as cw1, w2_dfb.wait() as cw2, w3_dfb.wait() as cw3:
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    with s0_dfb.wait() as v0, s1_dfb.wait() as v1, s2_dfb.wait() as v2, s3_dfb.wait() as v3:
                        with acc_dfb.reserve() as acc:
                            acc.store(cw0 * v0 + cw1 * v1 + cw2 * v2 + cw3 * v3)
                    with acc_dfb.wait() as x, out_dfb.reserve() as o:
                        o.store(x * ttl.math.sigmoid(x))  # SiLU

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        if core_x > 0:
            with bnd_dfb.reserve() as blk:
                def recv(pipe):
                    xf = ttl.copy(pipe, blk); xf.wait()
                net.if_dst(recv)
        # Load weights once
        with w0_dfb.reserve() as blk:
            tx = ttl.copy(w0[0, 0:HIDDEN_TILES], blk); tx.wait()
        with w1_dfb.reserve() as blk:
            tx = ttl.copy(w1[0, 0:HIDDEN_TILES], blk); tx.wait()
        with w2_dfb.reserve() as blk:
            tx = ttl.copy(w2[0, 0:HIDDEN_TILES], blk); tx.wait()
        with w3_dfb.reserve() as blk:
            tx = ttl.copy(w3[0, 0:HIDDEN_TILES], blk); tx.wait()
        # Stream sequence tiles, pipe last tile's input to next core
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                with s0_dfb.reserve() as blk:
                    tx = ttl.copy(s0[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()
                    if local_t == tiles_per_core - 1:
                        if core_x < N_CONV_CORES - 1:
                            def send(pipe):
                                xf = ttl.copy(blk, pipe); xf.wait()
                            net.if_src(send)
                with s1_dfb.reserve() as blk:
                    tx = ttl.copy(s1[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()
                with s2_dfb.reserve() as blk:
                    tx = ttl.copy(s2[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()
                with s3_dfb.reserve() as blk:
                    tx = ttl.copy(s3[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        if core_x > 0:
            prev_tile = core_x * tiles_per_core - 1
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[prev_tile, 0:HIDDEN_TILES]); tx.wait()
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[tile_idx, 0:HIDDEN_TILES]); tx.wait()
```

From `engram_demo_ttlang.py`. Key patterns:
- **Forward pipe chain**: each core pipes its last input tile to the next core for boundary handling
- **Pipe + streaming combined**: boundary receive happens before the main loop, send happens on the last iteration
- **Weights held in scope**: `buffer_factor=1` weights loaded once, kept in scope via outer `with` block across entire streaming loop
- **SiLU activation**: `x * sigmoid(x)` fused in one expression

### Example 6: Full MD Force Kernel (Real-World, 28 DFBs, Factory Pattern)

Complete cell-list molecular dynamics force kernel from a validated simulation (10K atoms, 10K steps, 1.1ms/step). Computes erfc-damped Coulomb + LJ 12-6 forces for all atom pairs across 27 neighbor cells. Shows: factory function pattern for parameterizing kernels with runtime constants, 28 DFBs near the hardware limit, broadcast+transpose for pairwise distance matrices, Horner polynomial evaluation, PBC minimum image convention, and init-then-accumulate force accumulators.

```python
TILE = 32
N_NBR = 27

def make_force_kernel(c_n_dim, c_dim2):
    """Factory: captures cell-grid dimensions as compile-time constants."""

    # Physics constants captured by closure
    c_box = float(box_length)
    c_inv_box = 1.0 / float(box_length)
    c_half = 0.5
    c_lj_scale = 24.0
    c_alpha_sq = float(alpha * alpha)
    c_p_alpha = float(ERFC_P * alpha)
    c_two_a_sp = float(2.0 * alpha / np.sqrt(np.pi))
    c_a1 = float(ERFC_A1)
    c_a2 = float(-ERFC_A2)
    c_a3 = float(ERFC_A3)
    c_a4 = float(-ERFC_A4)
    c_a5 = float(ERFC_A5)

    @ttl.kernel(grid="auto")
    def cell_forces_kernel(own_px, own_py, own_pz, own_q,
                           self_mask, scaler,
                           fx_out, fy_out, fz_out):
        grid_cols, _ = ttl.grid_size(dims=2)
        n_cells = own_px.shape[0] // TILE
        cells_per_core = -(-n_cells // grid_cols)

        ox_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        oy_cb = ttl.make_dataflow_buffer_like(own_py, shape=(1, 1), buffer_factor=2)
        oz_cb = ttl.make_dataflow_buffer_like(own_pz, shape=(1, 1), buffer_factor=2)
        oq_cb = ttl.make_dataflow_buffer_like(own_q, shape=(1, 1), buffer_factor=2)
        ex_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        ey_cb = ttl.make_dataflow_buffer_like(own_py, shape=(1, 1), buffer_factor=2)
        ez_cb = ttl.make_dataflow_buffer_like(own_pz, shape=(1, 1), buffer_factor=2)
        eq_cb = ttl.make_dataflow_buffer_like(own_q, shape=(1, 1), buffer_factor=2)
        sm_cb = ttl.make_dataflow_buffer_like(self_mask, shape=(1, 1), buffer_factor=2)
        sc_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)

        ba_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        tr_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        bb_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        r2_tmp = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)

        r2_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        qq_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        dx_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        dy_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        dz_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)

        fm_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        ft_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        fr_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)

        ax_cb = ttl.make_dataflow_buffer_like(fx_out, shape=(1, 1), buffer_factor=2)
        ay_cb = ttl.make_dataflow_buffer_like(fy_out, shape=(1, 1), buffer_factor=2)
        az_cb = ttl.make_dataflow_buffer_like(fz_out, shape=(1, 1), buffer_factor=2)

        fxo_cb = ttl.make_dataflow_buffer_like(fx_out, shape=(1, 1), buffer_factor=2)
        fyo_cb = ttl.make_dataflow_buffer_like(fy_out, shape=(1, 1), buffer_factor=2)
        fzo_cb = ttl.make_dataflow_buffer_like(fz_out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_c in range(cells_per_core):
                cell_id = core_x * cells_per_core + local_c
                if cell_id < n_cells:
                    with ox_cb.wait() as ox, oy_cb.wait() as oy, oz_cb.wait() as oz, oq_cb.wait() as oq:
                        with sc_cb.wait() as sc:
                            for nbr_i in range(N_NBR):
                                with ex_cb.wait() as ex, ey_cb.wait() as ey, ez_cb.wait() as ez, eq_cb.wait() as eq, sm_cb.wait() as sm:
                                    # PBC pairwise x-distances via broadcast+transpose
                                    with ba_cb.reserve() as ba:
                                        ba.store(ttl.math.broadcast(ox, dims=[1]))
                                    with tr_cb.reserve() as tr:
                                        tr.store(ttl.transpose(ex))
                                    with tr_cb.wait() as trv, bb_cb.reserve() as bb:
                                        bb.store(ttl.math.broadcast(trv, dims=[0]))
                                    with ba_cb.wait() as bav, bb_cb.wait() as bbv:
                                        dx_raw = bav - bbv
                                        dx_pbc = dx_raw - ttl.math.fill(bav, c_box) * ttl.math.floor(dx_raw * ttl.math.fill(bav, c_inv_box) + ttl.math.fill(bav, c_half))
                                        with r2_tmp.reserve() as r2o:
                                            r2o.store(dx_pbc * dx_pbc)
                                        with dx_cb.reserve() as dxo:
                                            dxo.store(dx_pbc)

                                    # PBC pairwise y-distances
                                    with ba_cb.reserve() as ba:
                                        ba.store(ttl.math.broadcast(oy, dims=[1]))
                                    with tr_cb.reserve() as tr:
                                        tr.store(ttl.transpose(ey))
                                    with tr_cb.wait() as trv, bb_cb.reserve() as bb:
                                        bb.store(ttl.math.broadcast(trv, dims=[0]))
                                    with ba_cb.wait() as bav, bb_cb.wait() as bbv, r2_tmp.wait() as r2p:
                                        dy_raw = bav - bbv
                                        dy_pbc = dy_raw - ttl.math.fill(bav, c_box) * ttl.math.floor(dy_raw * ttl.math.fill(bav, c_inv_box) + ttl.math.fill(bav, c_half))
                                        with r2_tmp.reserve() as r2o:
                                            r2o.store(r2p + dy_pbc * dy_pbc)
                                        with dy_cb.reserve() as dyo:
                                            dyo.store(dy_pbc)

                                    # PBC pairwise z-distances (adds self-exclusion mask to r2)
                                    with ba_cb.reserve() as ba:
                                        ba.store(ttl.math.broadcast(oz, dims=[1]))
                                    with tr_cb.reserve() as tr:
                                        tr.store(ttl.transpose(ez))
                                    with tr_cb.wait() as trv, bb_cb.reserve() as bb:
                                        bb.store(ttl.math.broadcast(trv, dims=[0]))
                                    with ba_cb.wait() as bav, bb_cb.wait() as bbv, r2_tmp.wait() as r2p:
                                        dz_raw = bav - bbv
                                        dz_pbc = dz_raw - ttl.math.fill(bav, c_box) * ttl.math.floor(dz_raw * ttl.math.fill(bav, c_inv_box) + ttl.math.fill(bav, c_half))
                                        with r2_cb.reserve() as r2o:
                                            r2o.store(r2p + dz_pbc * dz_pbc + sm)
                                        with dz_cb.reserve() as dzo:
                                            dzo.store(dz_pbc)

                                    # Charge products via broadcast+transpose
                                    with ba_cb.reserve() as ba:
                                        ba.store(ttl.math.broadcast(oq, dims=[1]))
                                    with tr_cb.reserve() as tr:
                                        tr.store(ttl.transpose(eq))
                                    with tr_cb.wait() as trv, bb_cb.reserve() as bb:
                                        bb.store(ttl.math.broadcast(trv, dims=[0]))
                                    with ba_cb.wait() as bav, bb_cb.wait() as bbv, qq_cb.reserve() as qqo:
                                        qqo.store(bav * bbv)

                                    # erfc-damped Coulomb + LJ 12-6 force magnitudes
                                    with r2_cb.wait() as r2, qq_cb.wait() as qq:
                                        r_inv = ttl.math.rsqrt(r2)
                                        r2_inv = ttl.math.recip(r2)
                                        r_val = r2 * r_inv
                                        t = ttl.math.recip(r_inv * r_inv * r2 + ttl.math.fill(r2, c_p_alpha) * r_val)
                                        poly = t * (ttl.math.fill(r2, c_a1) + t * (ttl.math.neg(ttl.math.fill(r2, c_a2)) + t * (ttl.math.fill(r2, c_a3) + t * (ttl.math.neg(ttl.math.fill(r2, c_a4)) + t * ttl.math.fill(r2, c_a5)))))
                                        exp_neg = ttl.math.exp(ttl.math.neg(ttl.math.fill(r2, c_alpha_sq) * r2))
                                        erfc_val = poly * exp_neg
                                        with ft_cb.reserve() as coul:
                                            coul.store(qq * (erfc_val * r2_inv + ttl.math.fill(r2, c_two_a_sp) * exp_neg * r_inv) * r_inv)
                                        r2_inv2 = ttl.math.recip(r2)
                                        r4_inv = r2_inv2 * r2_inv2
                                        r6_inv = r4_inv * r2_inv2
                                        r12_inv = r6_inv * r6_inv
                                        with fr_cb.reserve() as lj:
                                            lj.store(ttl.math.fill(r2, c_lj_scale) * r2_inv2 * (r12_inv + r12_inv - r6_inv))

                                    with ft_cb.wait() as fc, fr_cb.wait() as fl:
                                        with fm_cb.reserve() as fmo:
                                            fmo.store(fl + fc)

                                    # Project onto displacements, reduce rows, accumulate per-axis
                                    with fm_cb.wait() as fm:
                                        with dx_cb.wait() as dxv:
                                            with ft_cb.reserve() as ft:
                                                ft.store(fm * dxv)
                                            with ft_cb.wait() as ftv, fr_cb.reserve() as fr:
                                                fr.store(ttl.math.reduce_sum(ftv, sc, dims=[1]))
                                            if nbr_i == 0:
                                                with fr_cb.wait() as frv, ax_cb.reserve() as ax:
                                                    ax.store(frv)
                                            else:
                                                with fr_cb.wait() as frv, ax_cb.wait() as prev:
                                                    with ax_cb.reserve() as ax:
                                                        ax.store(prev + frv)
                                        with dy_cb.wait() as dyv:
                                            with ft_cb.reserve() as ft:
                                                ft.store(fm * dyv)
                                            with ft_cb.wait() as ftv, fr_cb.reserve() as fr:
                                                fr.store(ttl.math.reduce_sum(ftv, sc, dims=[1]))
                                            if nbr_i == 0:
                                                with fr_cb.wait() as frv, ay_cb.reserve() as ay:
                                                    ay.store(frv)
                                            else:
                                                with fr_cb.wait() as frv, ay_cb.wait() as prev:
                                                    with ay_cb.reserve() as ay:
                                                        ay.store(prev + frv)
                                        with dz_cb.wait() as dzv:
                                            with ft_cb.reserve() as ft:
                                                ft.store(fm * dzv)
                                            with ft_cb.wait() as ftv, fr_cb.reserve() as fr:
                                                fr.store(ttl.math.reduce_sum(ftv, sc, dims=[1]))
                                            if nbr_i == 0:
                                                with fr_cb.wait() as frv, az_cb.reserve() as az:
                                                    az.store(frv)
                                            else:
                                                with fr_cb.wait() as frv, az_cb.wait() as prev:
                                                    with az_cb.reserve() as az:
                                                        az.store(prev + frv)

                            with ax_cb.wait() as fx, fxo_cb.reserve() as fxo:
                                fxo.store(fx)
                            with ay_cb.wait() as fy, fyo_cb.reserve() as fyo:
                                fyo.store(fy)
                            with az_cb.wait() as fz, fzo_cb.reserve() as fzo:
                                fzo.store(fz)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for local_c in range(cells_per_core):
                cell_id = core_x * cells_per_core + local_c
                if cell_id < n_cells:
                    with ox_cb.reserve() as blk:
                        tx = ttl.copy(own_px[cell_id, 0], blk); tx.wait()
                    with oy_cb.reserve() as blk:
                        tx = ttl.copy(own_py[cell_id, 0], blk); tx.wait()
                    with oz_cb.reserve() as blk:
                        tx = ttl.copy(own_pz[cell_id, 0], blk); tx.wait()
                    with oq_cb.reserve() as blk:
                        tx = ttl.copy(own_q[cell_id, 0], blk); tx.wait()
                    with sc_cb.reserve() as blk:
                        tx = ttl.copy(scaler[0, 0], blk); tx.wait()
                    # Compute neighbor cell IDs from 3D grid coordinates
                    cx = cell_id // c_dim2
                    cy = (cell_id // c_n_dim) % c_n_dim
                    cz = cell_id % c_n_dim
                    for nbr in range(N_NBR):
                        off_dx = (nbr // 9) - 1
                        off_dy = ((nbr // 3) % 3) - 1
                        off_dz = (nbr % 3) - 1
                        nbr_cell = ((cx + off_dx + c_n_dim) % c_n_dim) * c_dim2 + ((cy + off_dy + c_n_dim) % c_n_dim) * c_n_dim + ((cz + off_dz + c_n_dim) % c_n_dim)
                        with ex_cb.reserve() as blk:
                            tx = ttl.copy(own_px[nbr_cell, 0], blk); tx.wait()
                        with ey_cb.reserve() as blk:
                            tx = ttl.copy(own_py[nbr_cell, 0], blk); tx.wait()
                        with ez_cb.reserve() as blk:
                            tx = ttl.copy(own_pz[nbr_cell, 0], blk); tx.wait()
                        with eq_cb.reserve() as blk:
                            tx = ttl.copy(own_q[nbr_cell, 0], blk); tx.wait()
                        with sm_cb.reserve() as blk:
                            tx = ttl.copy(self_mask[cell_id * N_NBR + nbr, 0], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_c in range(cells_per_core):
                cell_id = core_x * cells_per_core + local_c
                if cell_id < n_cells:
                    with fxo_cb.wait() as blk:
                        tx = ttl.copy(blk, fx_out[cell_id, 0]); tx.wait()
                    with fyo_cb.wait() as blk:
                        tx = ttl.copy(blk, fy_out[cell_id, 0]); tx.wait()
                    with fzo_cb.wait() as blk:
                        tx = ttl.copy(blk, fz_out[cell_id, 0]); tx.wait()

    return cell_forces_kernel

# Usage: kernel is built with runtime cell-grid parameters, then called repeatedly
cell_forces_kernel = make_force_kernel(c_n_dim=n_cells_dim, c_dim2=n_cells_dim**2)
cell_forces_kernel(tt_px, tt_py, tt_pz, tt_q, tt_masks, tt_scaler, tt_fx, tt_fy, tt_fz)
```

From `md_cell_list.py` (validated: 10K atoms, 10K steps, 1.1ms/step). Key patterns:
- **Factory function**: `make_force_kernel(c_n_dim, c_dim2)` captures cell-grid dimensions and physics constants as closure variables. The returned kernel is called many times per MD step with different positions.
- **28 DFBs** near the 32-DFB hardware limit: own cell (4), neighbor cell (5), geometry intermediates (9), force intermediates (3), xyz accumulators (3), output (3), scaler (1).
- **Pairwise distance via broadcast+transpose**: `broadcast(ox, dims=[1])` expands column vector to NxN, `transpose(ex)` + `broadcast(dims=[0])` expands row vector. Subtraction gives all-pairs distance matrix in one tile.
- **PBC minimum image**: `dx - box * floor(dx/box + 0.5)` with `ttl.math.fill` for scalar constants.
- **Self-exclusion mask**: Added directly to r2 so self-pairs and empty slots get r2~1e6, making forces vanish naturally.
- **Horner polynomial for erfc**: Nested multiply-add chain, 20+ fused elementwise ops in a single `with` block.
- **Init-then-accumulate** (CB threading Rule 3): `nbr_i == 0` initializes ax/ay/az via `reserve()`, subsequent neighbors do `wait()` + `reserve()` self-cycle.
- **Computed neighbor indices in dm_read**: 3D grid coordinates derived from flat cell_id, neighbor offsets applied with PBC wrapping. No pre-built index table needed.
- **Deeply nested scoping**: Own cell data held in scope across all 27 neighbor iterations (single `wait()`, reused 27 times).
