---
description: Import and translate a CUDA, Triton, PyTorch kernel, or TTNN program to TT-Lang DSL
argument-hint: <kernel-file-or-code>
---

## Tools Available

NOTE: these tools are already in PATH. You do not need to find them in a relative directory, you can invoke them directly from any directory.

NOTE: flags on run-test.sh must come before file argument. You can use --help if unsure on how to use.

NOTE: run-test.sh will copy the file. You do not need to copy the test file each time.

```bash
run-test.sh /path/to/kernel.py    # Run kernel on VM simulator (ONLY way to test)
copy-file.sh /path/to/file.py     # Copy a file to the VM
```

**Reading VM logs (output is saved, not streamed):**
```bash
limactl shell ttsim -- cat /tmp/ttlang_test_output.log        # Full log
limactl shell ttsim -- tail -100 /tmp/ttlang_test_output.log  # Last 100 lines
limactl shell ttsim -- grep -i "error" /tmp/ttlang_test_output.log
limactl shell ttsim -- cat /tmp/ttlang_initial.mlir           # Initial MLIR
limactl shell ttsim -- cat /tmp/ttlang_final.mlir             # Final MLIR
```

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
        inp = input_cb.wait()
        b = bias_cb.wait()
        o = out_cb.reserve()
        # All ops fuse into one compute body
        result = ttl.math.relu(ttl.math.exp(inp) + b)
        o.store(result)
        # ... pop/push ...
```

**When fusing TTNN ops:**
1. Identify the sequence of ops to fuse
2. Create one CB per input tensor
3. Chain operations in a single compute function
4. TT-Lang will generate optimized fused code

## TT-Lang Programming Model

### Kernel Structure

Every TT-Lang kernel has exactly three threads that run concurrently:
1. **Compute thread** (`@ttl.compute()`): Math operations on tiles in L1
2. **Reader thread** (`@ttl.datamovement()`): Loads data from DRAM to circular buffers
3. **Writer thread** (`@ttl.datamovement()`): Writes data from circular buffers to DRAM

These threads synchronize via **circular buffers** (CBs).

### Basic Kernel Template

```python
import ttl

@ttl.kernel(grid=(1, 1))
def add_kernel(lhs, rhs, out):
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with lhs_cb.wait() as l, rhs_cb.wait() as r, out_cb.reserve() as o:
            o.store(l + r)

    @ttl.datamovement()
    def dm_read():
        with lhs_cb.reserve() as blk:
            tx = ttl.copy(lhs[0, 0], blk)
            tx.wait()
        with rhs_cb.reserve() as blk:
            tx = ttl.copy(rhs[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
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
    with input1_cb.wait() as a, input2_cb.wait() as b:
        with output_cb.reserve() as o:
            result = a + b
            o.store(result)
    # pop/push happens automatically at end of with block

@ttl.datamovement()
def dm_read():
    with input1_cb.reserve() as blk:
        tx = ttl.copy(input1[0, 0], blk)
        tx.wait()
    # push happens automatically
```

### Circular Buffer API Reference

```python
# Create a circular buffer
cb = ttl.make_circular_buffer_like(
    tensor,           # TTNN tensor to inherit dtype/layout from
    shape=(R, C),     # Block size in tiles (e.g., (2, 2) = 4 tiles per block)
    buffer_factor=2   # Number of blocks in CB (2 = double buffering)
)

# Consumer operations (compute thread consumes data)
blk = cb.wait()       # Block until data available, returns block
cb.pop()              # Release block back to producer

# Producer operations (datamovement thread produces data)
blk = cb.reserve()    # Block until space available, returns block
cb.push()             # Signal data is ready for consumer

# Context manager (preferred - auto pop/push)
with cb.wait() as blk:      # For consumers
    # use blk...
with cb.reserve() as blk:   # For producers
    # fill blk...

# Block operations
blk.store(expr)             # Store result of expression into block
```

**CB Shape = Block Size:** The `shape=(R, C)` parameter defines the **block size** in tiles. A block is the unit of data transferred between threads. For tensors larger than one block, use **loops** to iterate over multiple blocks:

```python
# 128x128 tensor = 4x4 tiles, process in 2x2 blocks (4 iterations)
cb = ttl.make_circular_buffer_like(tensor, shape=(2, 2), buffer_factor=2)

@ttl.datamovement()
def dm_read():
    for row in range(2):      # 2 row-blocks
        for col in range(2):  # 2 col-blocks
            with cb.reserve() as blk:
                tx = ttl.copy(tensor[row*2:(row+1)*2, col*2:(col+1)*2], blk)
                tx.wait()

@ttl.compute()
def compute():
    for _ in range(4):  # Must match total iterations in dm_read
        with cb.wait() as blk, out_cb.reserve() as o:
            o.store(ttl.math.exp(blk))
```

## Available Operations

### Binary Operators

```python
result = a + b      # Element-wise addition
result = a - b      # Element-wise subtraction
result = a * b      # Element-wise multiplication
result = a / b      # Element-wise division
# NOTE: a @ b does NOT work! Use ttl.math.matmul() instead (see below)
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
```

### Matrix Multiplication (IMPORTANT: Different semantics!)

```python
# ttl.math.matmul is ACCUMULATING: C += A @ B
# The third argument is both the accumulator AND output
result = ttl.math.matmul(a, b, c)  # c += a @ b, returns updated c

# Example usage:
with a_cb.wait() as a_tile, b_cb.wait() as b_tile, c_cb.reserve() as c_out:
    result = ttl.math.matmul(a_tile, b_tile, c_out)
    c_out.store(result)
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
# Takes input and output blocks, works with multi-tile CBs
with inp_cb.wait() as x, out_cb.reserve() as o:
    result = ttl.transpose(x, o)
    o.store(result)
```

**Non-square example:** For 4x2 tiles → 2x4 tiles:
```python
inp_cb = ttl.make_circular_buffer_like(inp, shape=(4, 2), buffer_factor=2)
out_cb = ttl.make_circular_buffer_like(out, shape=(2, 4), buffer_factor=2)  # Swapped!
```

### Reductions (require scaler tensor)

```python
# Reductions are in ttl.math and need a "scaler" tensor (1x1 CB of all 1.0s)
# dims=[0] = row reduction, dims=[1] = col reduction, dims=[0, 1] = scalar

# Scaler: 32x32 tile of 1.0s in a 1x1 CB
scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)

with inp_cb.wait() as i, scaler_cb.wait() as s, out_cb.reserve() as o:
    # Scalar reduction (sum/max entire CB -> single value in output [0,0])
    result = ttl.math.reduce_sum(i, s, o, dims=[0, 1])
    result = ttl.math.reduce_max(i, s, o, dims=[0, 1])

    # Row reduction (reduce across rows)
    result = ttl.math.reduce_sum(i, s, o, dims=[0])

    # Column reduction (reduce across columns)
    result = ttl.math.reduce_sum(i, s, o, dims=[1])

    o.store(result)
```

**IMPORTANT - Dimension semantics differ from PyTorch:**
- `dims=[0]` for reduce gives **per-row results** (stored in column 0) - output shape [M, 1]
- `dims=[1]` for reduce gives **per-column results** (stored in row 0) - output shape [1, N]

In PyTorch, `dim=0` means "reduce along dimension 0" (collapse rows). In TT-Lang, `dims=[0]` means "keep dimension 0" (keep rows, collapse columns). The semantics are inverted.

**Multi-tile reduce:** Reduces across ALL tiles in the input CB. For example, a 4x1 tile input CB reduced with `dims=[0, 1]` produces a single scalar value (in a 1x1 output CB). The reduction sums all elements across all 4 tiles into position [0,0].

### Broadcast

```python
# Broadcast expands a smaller block to match a larger output shape
# dims=[0] = broadcast rows, dims=[1] = broadcast cols, dims=[0, 1] = broadcast scalar

with scalar_cb.wait() as s, out_cb.reserve() as o:
    # Broadcast 1x1 scalar to fill entire output block
    result = ttl.math.broadcast(s, o, dims=[0, 1])
    o.store(result)

with row_cb.wait() as r, out_cb.reserve() as o:
    # Broadcast 1xN row across M rows
    result = ttl.math.broadcast(r, o, dims=[0])
    o.store(result)

with col_cb.wait() as c, out_cb.reserve() as o:
    # Broadcast Mx1 column across N columns
    result = ttl.math.broadcast(c, o, dims=[1])
    o.store(result)
```

**IMPORTANT - Broadcast dimension semantics:**
- `dims=[1]` for broadcast **copies column 0 to all columns** - input (N, 1) -> output (N, M)
- `dims=[0]` for broadcast **copies row 0 to all rows** - input (1, M) -> output (N, M)

Note: Reduce and broadcast dims have complementary semantics. `dims=[0]` reduce produces a single column (per-row results), `dims=[1]` broadcast replicates that column across all columns.

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
    with input_cb.wait() as a, bias_cb.wait() as b, out_cb.reserve() as o:
        # All these ops fuse into one efficient compute body
        x = ttl.math.exp(a)
        y = x + b
        z = ttl.math.sigmoid(y)
        result = ttl.math.relu(z)
        o.store(result)
```

**Fusion with broadcast:** Broadcast can be fused as the first operation in a chain:
```python
with scalar_cb.wait() as s, other_cb.wait() as x, out_cb.reserve() as o:
    bcast = ttl.math.broadcast(s, o, dims=[0, 1])  # First op - OK to fuse
    result = bcast * x + ttl.math.exp(x)           # Continues fusion chain
    o.store(result)
```

**Limitation:** Ops that take CB arguments (matmul, reduce, transpose, broadcast) can only be fused if they are the **first** operation. After any of these, you must store and start a new fusion chain.

**When fusion fails:** Use sequential `with` blocks to break the chain - you do NOT need separate kernels:

```python
@ttl.compute()
def compute():
    # WRONG: Trying to fuse matmul result into another CB op
    # with a_cb.wait() as a, b_cb.wait() as b, out_cb.reserve() as o:
    #     m = ttl.math.matmul(a, b, o)
    #     result = ttl.math.reduce_sum(m, ...)  # FAILS - reduce after matmul

    # CORRECT: Break into two with blocks (still one kernel!)
    with a_cb.wait() as a, b_cb.wait() as b, intermediate_cb.reserve() as inter:
        m = ttl.math.matmul(a, b, inter)
        inter.store(m)

    with intermediate_cb.wait() as inter, scaler_cb.wait() as s, out_cb.reserve() as o:
        result = ttl.math.reduce_sum(inter, s, o, dims=[0, 1])
        o.store(result)
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

For tensors larger than 32x32, process multiple tiles. **Use multicore and loops:**

- **Multicore is encouraged** - use the whole chip for real workloads. Single-core kernels are fine for incremental development but won't deliver meaningful performance.
- **Loops are supported** in both compute and datamovement threads - use them to stream large tensors through smaller CBs.

### IMPORTANT: Match the User's Target Data Size

**If the user provides a specific model config or tensor shape, strive to support that size.** You can simplify to smaller tensors for initial testing and debugging, but the goal is a kernel that works on their actual data. Use loops and streaming to handle large inputs:

```python
# User wants to process 2048x2048 tensors (64x64 tiles)
# Don't shrink to 32x32 for testing - make it work at target size!

@ttl.kernel(grid=(8, 8))  # Use multicore
def large_tensor_kernel(inp, out):
    # Each core handles 8x8 tiles worth of data
    # But CB only holds 2x2 tiles at a time - stream through with loops

    inp_cb = ttl.make_circular_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)

    BLOCKS_PER_CORE = 4  # 8x8 tiles / 2x2 block = 4x4 = 16 blocks... adjust per core

    @ttl.compute()
    def compute():
        for _ in range(BLOCKS_PER_CORE):
            with inp_cb.wait() as i, out_cb.reserve() as o:
                o.store(ttl.math.exp(i))

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.core(dims=2)
        for block_idx in range(BLOCKS_PER_CORE):
            # Calculate tile coordinates for this core and block
            row = y * 8 + (block_idx // 4) * 2  # Example indexing
            col = x * 8 + (block_idx % 4) * 2
            with inp_cb.reserve() as blk:
                tx = ttl.copy(inp[row:row+2, col:col+2], blk)
                tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.core(dims=2)
        for block_idx in range(BLOCKS_PER_CORE):
            row = y * 8 + (block_idx // 4) * 2
            col = x * 8 + (block_idx % 4) * 2
            with out_cb.wait() as blk:
                tx = ttl.copy(blk, out[row:row+2, col:col+2])
                tx.wait()
```

**Key streaming principles:**
1. **CB size is limited by L1** (~1.5MB per core) - you can't fit huge tensors
2. **Stream blocks through CBs** - read a block, process it, write it, repeat
3. **Loop counts must match** - compute iterations = dm_read iterations = dm_write iterations
4. **DRAM is large but slow** - keep data in L1 as long as possible, stream to avoid DRAM round-trips

## Pipes (Core-to-Core Communication)

**WARNING: Use pipes sparingly.** Pipes enable communication between cores but are error-prone and a common cause of hangs. Get your kernel working without pipes first, then add them only if needed for performance.

### Pipe API

```python
# Create a pipe: one source core to one destination core
pipe = ttl.Pipe(src=(source_x, source_y), dst=(dest_x, dest_y))

# Send data through pipe (in dm_write on source core)
tx = ttl.copy(blk, pipe)
tx.wait()

# Receive data from pipe (in dm_write on destination core)
tx = ttl.copy(pipe, blk)
tx.wait()
```

### Complete Example: Gather Pattern

This example from `test_full_reduce_bcast_matmul.py` shows workers sending results to a coordinator:

```python
@ttl.kernel(grid=(4, 1))
def gather_kernel(inp, out):
    # Create one pipe per worker -> coordinator
    pipe1 = ttl.Pipe(src=(1, 0), dst=(0, 0))
    pipe2 = ttl.Pipe(src=(2, 0), dst=(0, 0))
    pipe3 = ttl.Pipe(src=(3, 0), dst=(0, 0))

    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    gather_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=4)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        x, y = ttl.core(dims=2)
        if x == 0:
            # Coordinator: accumulate from gather_cb
            for _ in range(3):
                with gather_cb.wait() as g, out_cb.reserve() as o:
                    o.store(g)  # Simplified - real code would accumulate
        else:
            # Workers: process and send to gather_cb
            with inp_cb.wait() as i, gather_cb.reserve() as g:
                g.store(ttl.math.exp(i))

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.core(dims=2)
        with inp_cb.reserve() as blk:
            tx = ttl.copy(inp[y, x], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.core(dims=2)

        # Workers send their results via pipes
        if x == 1:
            with gather_cb.wait() as blk:
                tx = ttl.copy(blk, pipe1)
                tx.wait()
        elif x == 2:
            with gather_cb.wait() as blk:
                tx = ttl.copy(blk, pipe2)
                tx.wait()
        elif x == 3:
            with gather_cb.wait() as blk:
                tx = ttl.copy(blk, pipe3)
                tx.wait()

        # Coordinator receives from all pipes
        if x == 0:
            with gather_cb.reserve() as blk:
                tx = ttl.copy(pipe1, blk)
                tx.wait()
            with gather_cb.reserve() as blk:
                tx = ttl.copy(pipe2, blk)
                tx.wait()
            with gather_cb.reserve() as blk:
                tx = ttl.copy(pipe3, blk)
                tx.wait()

            # Write final output
            with out_cb.wait() as blk:
                tx = ttl.copy(blk, out[0, 0])
                tx.wait()
```

### Pipe Debugging Tips

- **Pipes cause hangs** when send/receive don't match - every `ttl.copy(blk, pipe)` needs a corresponding `ttl.copy(pipe, blk)`
- **IMPORTANT: Set a low timeout** when testing pipes for faster iteration
- **Start without pipes** - get single-core or independent multi-core working first
- **Add pipes incrementally** - test after adding each pipe
- **Kill zombie processes** if hung: `limactl shell ttsim -- pkill -9 python`

### Hardware Limits

- **32 CBs max** per core
- **~1.5MB L1** per core
- **~100MB total SRAM** across chip - utilize as much as possible for throughput
- **Tile size**: 32x32 elements = 2KB (bfloat16)

### Option 1: Large CB Shape (Single Core)

Larger CB shapes give better throughput. Aim for 4x4 or 8x8 if L1 allows:

```python
# 64x64 tensor = 2x2 tiles
@ttl.kernel(grid=(1, 1))
def multitile_kernel(lhs, rhs, out):
    # CB holds all 4 tiles - larger shapes better for throughput
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(2, 2), buffer_factor=2)
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape=(2, 2), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def compute():
        with lhs_cb.wait() as l, rhs_cb.wait() as r, out_cb.reserve() as o:
            o.store(l + r)  # Operates on all tiles

    @ttl.datamovement()
    def dm_read():
        with lhs_cb.reserve() as blk:
            tx = ttl.copy(lhs[0:2, 0:2], blk)  # Slice: rows 0-1, cols 0-1 (in tiles)
            tx.wait()
        # ... similar for rhs ...
```

### Option 2: Multicore (Parallel)

```python
# 256x256 tensor across 8x8 grid = 1 tile per core
@ttl.kernel(grid=(8, 8))
def multicore_kernel(lhs, rhs, out):
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with lhs_cb.wait() as l, rhs_cb.wait() as r:
            with out_cb.reserve() as o:
                result = l + r
                o.store(result)

    @ttl.datamovement()
    def dm_read():
        # Get this core's coordinates
        x, y = ttl.core(dims=2)  # x=column, y=row

        with lhs_cb.reserve() as blk:
            # Tensor indexing is [row, col] = [y, x]
            tx = ttl.copy(lhs[y, x], blk)
            tx.wait()

        with rhs_cb.reserve() as blk:
            tx = ttl.copy(rhs[y, x], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.core(dims=2)
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[y, x])
            tx.wait()

# Call: multicore_kernel(lhs, rhs, out)
```

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

with x_cb.wait() as x, scaler_cb.wait() as s:
    # 1. Find max for numerical stability
    with max_cb.reserve() as mx:
        mx.store(ttl.math.reduce_max(x, s, mx, dims=[0, 1]))

    # 2. Broadcast max back to full size
    with max_cb.wait() as mxv, bcast_cb.reserve() as mxb:
        mxb.store(ttl.math.broadcast(mxv, mxb, dims=[0, 1]))

    # 3. Compute exp(x - max) and sum
    with bcast_cb.wait() as max_bcast:
        shifted = x - max_bcast
        exp_shifted = ttl.math.exp(shifted)

        with sum_cb.reserve() as sm:
            sm.store(ttl.math.reduce_sum(exp_shifted, s, sm, dims=[0, 1]))

        # 4. Broadcast sum and divide
        with sum_cb.wait() as sumv, sum_bcast_cb.reserve() as smb:
            smb.store(ttl.math.broadcast(sumv, smb, dims=[0, 1]))

        with sum_bcast_cb.wait() as sum_bcast, out_cb.reserve() as o:
            o.store(ttl.math.exp(x - max_bcast) / sum_bcast)
```

### Key Principle

When you are re-writing a high level operation or kernel:
1. **What does this kernel or op do at a HW level?** Think about what's actually happening in the HW when this op runs
2. **What primitives do we have?** matmul, elementwise, DMA with indexing
3. **Build it from primitives.** A naive O(n²) loop that works is better than giving up. The goal is NOT performance! Just correctness. 
4. This is not a high level DSL like pytorch or ttnn, it's low level and you have explicit control over all of the HW, memory management, and synchronization. Do not think about direct mappings for high level ops and kernels, think about the best way to represent the kernel in tt-lang at the level it is designed to operate.

Even ops that DO exist may have different semantics (write in place, different numerical behavior). Always test to verify.

## Translation Guide: GPU → TT-Lang

### Concept Mapping

| GPU Concept | TT-Lang Equivalent |
|------------|-------------------|
| Thread block / workgroup | Grid of Tensix cores (`grid=(rows, cols)`) |
| Shared memory | L1 via circular buffers |
| Global memory | DRAM with DMA transfers |
| Warp/wave operations | Tile-level operations (32x32) |
| `__syncthreads()` | CB `wait()`/`push()` synchronization |
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
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    c_cb = ttl.make_circular_buffer_like(c, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_cb.wait() as av, b_cb.wait() as bv:
            with c_cb.reserve() as cv:
                result = av + bv  # Operates on entire 32x32 tile
                cv.store(result)

    @ttl.datamovement()
    def dm_read():
        with a_cb.reserve() as blk:
            tx = ttl.copy(a[0, 0], blk)
            tx.wait()
        with b_cb.reserve() as blk:
            tx = ttl.copy(b[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with c_cb.wait() as blk:
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
    x_cb = ttl.make_circular_buffer_like(x, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with x_cb.wait() as xv:
            with out_cb.reserve() as o:
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

**The VM is the ONLY place to test kernels. You MUST test every kernel you write.**

```
1. Write kernel to file
2. Run: run-test.sh /path/to/kernel.py
3. Read log: limactl shell ttsim -- tail -100 /tmp/ttlang_test_output.log
4. If errors: fix and go to step 2
5. If success: verify numerical output is correct
```

**IMPORTANT:**
- Exit code 0 does NOT mean success - always read the log
- The log can be thousands of lines - use `tail`, `head`, `grep` to navigate
- Look for: `AssertionError`, `Exception`, `error:`, `FAIL`, `mismatch`
- Never guess at fixes - always read the actual error message
- **IMPORTANT:** Set a low timeout for faster iteration - tests should execute in under 1 second. Hangs are common (especially with pipes or CB mismatches) and a low timeout helps detect them quickly.

**Handling Hangs:**
- If a kernel hangs, the most common cause is **CB mismatch** - every `wait()` needs a corresponding `push()` from producer, every `reserve()` needs a corresponding `pop()` from consumer
- Verify loop counts match between compute and datamovement threads
- Kill zombie processes on VM: `limactl shell ttsim -- pkill -9 python`

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
- Ops might take circular buffers as arguments
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
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with inp_cb.wait() as x:
            with out_cb.reserve() as o:
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
4. **Verify CB balance**: Every `wait()` needs `pop()`, every `reserve()` needs `push()`
5. **Read the log**: Always check `/tmp/ttlang_test_output.log` after each run
6. **Check MLIR**: Use `/tmp/ttlang_initial.mlir` and `/tmp/ttlang_final.mlir` for compiler issues

## Output

1. Save the translated TT-Lang kernel to a file
2. Run `run-test.sh` on the kernel and verify it works
3. Read the log and confirm numerical correctness
4. Report any TTNN ops used to fill gaps
5. Only mark complete after the kernel runs successfully on the VM

---

## Reference Examples

### Complete Example: Distributed Reduce-Broadcast-Matmul with Pipes

This is a complete working test from `test/python/test_full_reduce_bcast_matmul.py` showing multicore computation with pipes:

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Distributed reduce -> bcast -> matmul with multicore communication.

Each core independently:
1. Reduces its own A slice to a local scalar
2. Broadcasts that scalar to 4x4 tiles
3. Matmuls broadcasted value with B

Then coordinator gathers and sums all partial matmul results.

Grid: 4x1 (4 cores in a row)
"""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1

import ttl


COORDINATOR = 0
ROWS_PER_CORE = 2
COLS_PER_CORE = 2
NUM_WORKERS = 3


@ttl.kernel(grid=(4, 1))
def full_reduce_bcast_matmul_kernel(A, B, scaler, out):
    # Pipes for matmul result gather (workers -> coordinator)
    matmul_pipe1 = ttl.Pipe(src=(1, 0), dst=(0, 0))
    matmul_pipe2 = ttl.Pipe(src=(2, 0), dst=(0, 0))
    matmul_pipe3 = ttl.Pipe(src=(3, 0), dst=(0, 0))

    # Input CBs for reduce (A slice: 4 rows x 1 col of tiles)
    a_cb = ttl.make_circular_buffer_like(A, shape=(4, 1), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)

    # Input CB for matmul (B: 4x4 tiles)
    b_cb = ttl.make_circular_buffer_like(B, shape=(4, 4), buffer_factor=2)

    # Reduce intermediate CBs
    reduce_out_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    reduce_acc_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)

    # Broadcast CB (4x4 output)
    bcast_out_cb = ttl.make_circular_buffer_like(out, shape=(4, 4), buffer_factor=2)

    # Matmul CBs (4x4 tiles)
    matmul_out_cb = ttl.make_circular_buffer_like(out, shape=(4, 4), buffer_factor=2)
    matmul_gather_cb = ttl.make_circular_buffer_like(out, shape=(4, 4), buffer_factor=6)
    matmul_acc_cb = ttl.make_circular_buffer_like(out, shape=(4, 4), buffer_factor=2)

    # Output CB
    out_cb = ttl.make_circular_buffer_like(out, shape=(4, 4), buffer_factor=2)

    @ttl.compute()
    def compute():
        x, y = ttl.core(dims=2)

        # === Stage 1: Local reduce of A slices (all cores) ===
        blocks_per_core = ROWS_PER_CORE * COLS_PER_CORE

        with scaler_cb.wait() as s:
            # First block: reduce and copy to accumulator
            with a_cb.wait() as a, reduce_out_cb.reserve() as r:
                reduced = ttl.math.reduce_sum(a, s, r, dims=[0, 1])
                r.store(reduced)
            with reduce_out_cb.wait() as t, reduce_acc_cb.reserve() as acc:
                acc.store(ttl.math.abs(t))

            # Additional blocks: reduce and accumulate
            for _ in range(blocks_per_core - 1):
                with a_cb.wait() as a, reduce_out_cb.reserve() as r:
                    reduced = ttl.math.reduce_sum(a, s, r, dims=[0, 1])
                    r.store(reduced)
                with reduce_out_cb.wait() as t, reduce_acc_cb.wait() as acc, reduce_acc_cb.reserve() as new_acc:
                    new_acc.store(acc + t)

        # === Stage 2: Broadcast local sum to 4x4 tiles (all cores) ===
        with reduce_acc_cb.wait() as local_sum, bcast_out_cb.reserve() as bout:
            broadcasted = ttl.math.broadcast(local_sum, bout, dims=[0, 1])
            bout.store(broadcasted)

        # === Stage 3: Matmul (4x4) @ (4x4) -> (4x4) (all cores) ===
        with bcast_out_cb.wait() as a_bcast, b_cb.wait() as b, matmul_out_cb.reserve() as c:
            result = ttl.math.matmul(a_bcast, b, c)
            c.store(result)

        # === Stage 4: Gather and accumulate matmul results ===
        if x == COORDINATOR:
            with matmul_out_cb.wait() as m0, matmul_acc_cb.reserve() as macc:
                macc.store(ttl.math.abs(m0))

            for _ in range(NUM_WORKERS):
                with matmul_gather_cb.wait() as m, matmul_acc_cb.wait() as acc, matmul_acc_cb.reserve() as new_acc:
                    new_acc.store(acc + m)

            with matmul_acc_cb.wait() as acc, out_cb.reserve() as final_out:
                final_out.store(ttl.math.abs(acc))
        else:
            with matmul_out_cb.wait() as mout, matmul_gather_cb.reserve() as mg:
                mg.store(ttl.math.abs(mout))

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.core(dims=2)

        with scaler_cb.reserve() as s_blk:
            tx = ttl.copy(scaler[0, 0], s_blk)
            tx.wait()

        for row in range(ROWS_PER_CORE):
            for col in range(COLS_PER_CORE):
                row_idx = row * 4
                col_idx = x * COLS_PER_CORE + col
                with a_cb.reserve() as a_blk:
                    tx = ttl.copy(A[row_idx:row_idx+4, col_idx:col_idx+1], a_blk)
                    tx.wait()

        with b_cb.reserve() as b_blk:
            tx = ttl.copy(B[0:4, 0:4], b_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.core(dims=2)

        # Workers send to coordinator
        if x == 1:
            with matmul_gather_cb.wait() as blk:
                tx = ttl.copy(blk, matmul_pipe1)
                tx.wait()
        elif x == 2:
            with matmul_gather_cb.wait() as blk:
                tx = ttl.copy(blk, matmul_pipe2)
                tx.wait()
        elif x == 3:
            with matmul_gather_cb.wait() as blk:
                tx = ttl.copy(blk, matmul_pipe3)
                tx.wait()

        # Coordinator receives and writes output
        if x == COORDINATOR:
            with matmul_gather_cb.reserve() as blk:
                tx = ttl.copy(matmul_pipe1, blk)
                tx.wait()
            with matmul_gather_cb.reserve() as blk:
                tx = ttl.copy(matmul_pipe2, blk)
                tx.wait()
            with matmul_gather_cb.reserve() as blk:
                tx = ttl.copy(matmul_pipe3, blk)
                tx.wait()

            with out_cb.wait() as o_blk:
                tx = ttl.copy(o_blk, out[0:4, 0:4])
                tx.wait()


def test_uniform_values(device):
    A_height = ROWS_PER_CORE * 4 * 32
    A_width = 4 * COLS_PER_CORE * 32

    A_torch = torch.full((A_height, A_width), 0.01, dtype=torch.bfloat16)
    B_torch = torch.full((128, 128), 0.01, dtype=torch.bfloat16)
    scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
    out_torch = torch.zeros((128, 128), dtype=torch.bfloat16)

    A = to_l1(A_torch, device)
    B = to_l1(B_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_l1(out_torch, device)

    # Expected: matmul(broadcast(sum(A)), B)
    global_sum = A_torch.float().sum()
    A_bcast = torch.full((128, 128), global_sum.item(), dtype=torch.float32)
    expected = torch.matmul(A_bcast, B_torch.float())

    full_reduce_bcast_matmul_kernel(A, B, scaler, out)
    result = ttnn.to_torch(out).float()

    assert_allclose(result[0, 0], expected[0, 0], rtol=0.15, atol=500)
```

---

### Full Working Example: Fused MLP
```python
@ttl.kernel(grid=(1, 1))
def fused_mlp_kernel(x, w_fc, w_proj, out):
    """
    Fused MLP: out = relu²(x @ w_fc) @ w_proj

    x: (4, 4) tiles
    w_fc: (4, 16) tiles
    w_proj: (16, 4) tiles
    out: (4, 4) tiles
    """
    SEQ_TILES, EMBD_TILES, MLP_TILES = 4, 4, 16

    x_cb = ttl.make_circular_buffer_like(x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2)
    w_fc_cb = ttl.make_circular_buffer_like(w_fc, shape=(EMBD_TILES, MLP_TILES), buffer_factor=2)
    w_proj_cb = ttl.make_circular_buffer_like(w_proj, shape=(MLP_TILES, EMBD_TILES), buffer_factor=2)
    mlp_hidden_cb = ttl.make_circular_buffer_like(w_fc, shape=(SEQ_TILES, MLP_TILES), buffer_factor=2)
    mlp_act_cb = ttl.make_circular_buffer_like(w_fc, shape=(SEQ_TILES, MLP_TILES), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        # Step 1: x @ w_fc -> mlp_hidden
        with x_cb.wait() as xv, w_fc_cb.wait() as wfc:
            with mlp_hidden_cb.reserve() as mh:
                result = ttl.math.matmul(xv, wfc, mh)
                mh.store(result)

        # Step 2: relu²(mlp_hidden) -> mlp_act
        with mlp_hidden_cb.wait() as mhv, mlp_act_cb.reserve() as ma:
            relu_x = ttl.math.relu(mhv)
            ma.store(relu_x * relu_x)

        # Step 3: mlp_act @ w_proj -> out
        with mlp_act_cb.wait() as mav, w_proj_cb.wait() as wproj:
            with out_cb.reserve() as o:
                result = ttl.math.matmul(mav, wproj, o)
                o.store(result)

    @ttl.datamovement()
    def dm_read():
        with x_cb.reserve() as blk:
            tx = ttl.copy(x[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with w_fc_cb.reserve() as blk:
            tx = ttl.copy(w_fc[0:EMBD_TILES, 0:MLP_TILES], blk)
            tx.wait()
        with w_proj_cb.reserve() as blk:
            tx = ttl.copy(w_proj[0:MLP_TILES, 0:EMBD_TILES], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:EMBD_TILES])
            tx.wait()
```

---

## Pattern 3: Keeping Values in Scope for Residuals

Use **nested `with` blocks** to keep intermediate values accessible.

```python
@ttl.compute()
def compute():
    with scaler_cb.wait() as sc:
        # Keep x in scope for residual at the end
        with x_cb.wait() as xv:
            # ... many operations using xv ...

            with some_intermediate_cb.wait() as intermediate:
                # ... more operations ...

                # xv is STILL in scope here!
                with out_cb.reserve() as o:
                    o.store(intermediate + xv)  # Residual connection works
```

### Full Working Example: Post-Attention Block with Residuals
```python
@ttl.kernel(grid=(1, 1))
def fused_block_kernel(attn_concat, x, wo, ln2_w, w_fc, w_proj, scaler, out):
    """
    Fused: attn_proj -> residual1 -> RMSNorm -> MLP -> residual2
    """
    SEQ_TILES, EMBD_TILES, MLP_TILES = 4, 4, 16
    MLP_CHUNK_TILES, NUM_MLP_CHUNKS = 4, 4

    # Input CBs - buffer_factor=1 for single-use weights
    attn_cb = ttl.make_circular_buffer_like(attn_concat, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=1)
    x_cb = ttl.make_circular_buffer_like(x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=1)
    wo_cb = ttl.make_circular_buffer_like(wo, shape=(EMBD_TILES, EMBD_TILES), buffer_factor=1)
    ln2_w_cb = ttl.make_circular_buffer_like(ln2_w, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=1)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=1)

    # Streaming MLP weight chunks
    w_fc_chunk_cb = ttl.make_circular_buffer_like(w_fc, shape=(EMBD_TILES, MLP_CHUNK_TILES), buffer_factor=2)
    w_proj_chunk_cb = ttl.make_circular_buffer_like(w_proj, shape=(MLP_CHUNK_TILES, EMBD_TILES), buffer_factor=2)

    # Intermediate CBs
    act_cb = ttl.make_circular_buffer_like(x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2)
    hidden1_cb = ttl.make_circular_buffer_like(x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2)
    ln2_out_cb = ttl.make_circular_buffer_like(x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2)
    mlp_chunk_cb = ttl.make_circular_buffer_like(w_fc, shape=(SEQ_TILES, MLP_CHUNK_TILES), buffer_factor=2)
    mlp_acc_cb = ttl.make_circular_buffer_like(x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2)
    partial_cb = ttl.make_circular_buffer_like(x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2)
    reduce_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    bcast_cb = ttl.make_circular_buffer_like(x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=1)
    out_cb = ttl.make_circular_buffer_like(out, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=1)

    @ttl.compute()
    def compute():
        with scaler_cb.wait() as sc:
            # Step 1: attn_concat @ Wo -> attn_proj
            with attn_cb.wait() as attn, wo_cb.wait() as wo:
                with act_cb.reserve() as ap:
                    result = ttl.math.matmul(attn, wo, ap)
                    ap.store(result)

            # Step 2: attn_proj + x -> hidden1 (Residual 1)
            with act_cb.wait() as ap, x_cb.wait() as xv:
                with hidden1_cb.reserve() as h1:
                    h1.store(ap + xv)

            # Keep hidden1 in scope for residual 2!
            with hidden1_cb.wait() as h1v, ln2_w_cb.wait() as ln2_wv:
                # Step 3: RMSNorm
                with act_cb.reserve() as sq:
                    sq.store(h1v * h1v)
                with act_cb.wait() as sqv, reduce_cb.reserve() as red:
                    total = ttl.math.reduce_sum(sqv, sc, red, dims=[0, 1])
                    red.store(total)
                with reduce_cb.wait() as sumv, reduce_cb.reserve() as rsq:
                    rsq.store(ttl.math.rsqrt(sumv))
                with reduce_cb.wait() as rsqv, bcast_cb.reserve() as bc:
                    bc.store(ttl.math.broadcast(rsqv, bc, dims=[0, 1]))
                with bcast_cb.wait() as rsqrt_bcast, ln2_out_cb.reserve() as ln2:
                    normalized = h1v * rsqrt_bcast
                    ln2.store(normalized * ln2_wv)

                # Step 4: Streaming MLP (see Pattern 4 below)
                with ln2_out_cb.wait() as ln2v:
                    # First chunk
                    with w_fc_chunk_cb.wait() as wfc, mlp_chunk_cb.reserve() as mh:
                        result = ttl.math.matmul(ln2v, wfc, mh)
                        mh.store(result)
                    with mlp_chunk_cb.wait() as mhv, mlp_chunk_cb.reserve() as ma:
                        relu_x = ttl.math.relu(mhv)
                        ma.store(relu_x * relu_x)
                    with mlp_chunk_cb.wait() as mav, w_proj_chunk_cb.wait() as wpr:
                        with mlp_acc_cb.reserve() as acc:
                            result = ttl.math.matmul(mav, wpr, acc)
                            acc.store(result)

                    # Remaining chunks with accumulation
                    for _ in range(NUM_MLP_CHUNKS - 1):
                        with w_fc_chunk_cb.wait() as wfc, mlp_chunk_cb.reserve() as mh:
                            result = ttl.math.matmul(ln2v, wfc, mh)
                            mh.store(result)
                        with mlp_chunk_cb.wait() as mhv, mlp_chunk_cb.reserve() as ma:
                            relu_x = ttl.math.relu(mhv)
                            ma.store(relu_x * relu_x)
                        with mlp_chunk_cb.wait() as mav, w_proj_chunk_cb.wait() as wpr:
                            with partial_cb.reserve() as part:
                                result = ttl.math.matmul(mav, wpr, part)
                                part.store(result)
                        with partial_cb.wait() as partv, mlp_acc_cb.wait() as acc:
                            with mlp_acc_cb.reserve() as new_acc:
                                new_acc.store(acc + partv)

                # Step 5: Residual 2 - h1v still in scope!
                with mlp_acc_cb.wait() as mlp_out, out_cb.reserve() as o:
                    o.store(mlp_out + h1v)

    @ttl.datamovement()
    def dm_read():
        # Load fixed inputs once
        with attn_cb.reserve() as blk:
            tx = ttl.copy(attn_concat[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with x_cb.reserve() as blk:
            tx = ttl.copy(x[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with wo_cb.reserve() as blk:
            tx = ttl.copy(wo[0:EMBD_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with ln2_w_cb.reserve() as blk:
            tx = ttl.copy(ln2_w[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with scaler_cb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

        # Stream MLP weight chunks - loop syncs with compute
        for chunk in range(NUM_MLP_CHUNKS):
            col_start = chunk * MLP_CHUNK_TILES
            col_end = col_start + MLP_CHUNK_TILES
            with w_fc_chunk_cb.reserve() as blk:
                tx = ttl.copy(w_fc[0:EMBD_TILES, col_start:col_end], blk)
                tx.wait()
            with w_proj_chunk_cb.reserve() as blk:
                tx = ttl.copy(w_proj[col_start:col_end, 0:EMBD_TILES], blk)
                tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:EMBD_TILES])
            tx.wait()
```

---

## Pattern 4: Streaming Large Tensors

When tensors don't fit in L1, stream them through in chunks using synchronized loops.

**Key insight**: Loop count in compute must match loop count in datamovement.

```python
# Config
MLP_TILES = 16           # Full size
MLP_CHUNK_TILES = 4      # Chunk size
NUM_MLP_CHUNKS = 4       # 16 / 4 = 4 chunks

@ttl.kernel(grid=(1, 1))
def streaming_mlp_kernel(x, w_fc, w_proj, out):
    """Stream large MLP weights through small CBs."""
    SEQ_TILES, EMBD_TILES = 4, 4

    # Input stays in L1 (loaded once)
    x_cb = ttl.make_circular_buffer_like(x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2)

    # Streaming weight CBs - SMALL, reused per chunk
    w_fc_chunk_cb = ttl.make_circular_buffer_like(w_fc, shape=(EMBD_TILES, MLP_CHUNK_TILES), buffer_factor=2)
    w_proj_chunk_cb = ttl.make_circular_buffer_like(w_proj, shape=(MLP_CHUNK_TILES, EMBD_TILES), buffer_factor=2)

    # MLP hidden chunk
    mlp_chunk_cb = ttl.make_circular_buffer_like(w_fc, shape=(SEQ_TILES, MLP_CHUNK_TILES), buffer_factor=2)

    # Output accumulator
    out_acc_cb = ttl.make_circular_buffer_like(out, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2)
    partial_cb = ttl.make_circular_buffer_like(out, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        with x_cb.wait() as xv:
            # First chunk - initialize accumulator
            with w_fc_chunk_cb.wait() as wfc, mlp_chunk_cb.reserve() as mh:
                result = ttl.math.matmul(xv, wfc, mh)
                mh.store(result)
            with mlp_chunk_cb.wait() as mhv, mlp_chunk_cb.reserve() as ma:
                relu_x = ttl.math.relu(mhv)
                ma.store(relu_x * relu_x)
            with mlp_chunk_cb.wait() as mav, w_proj_chunk_cb.wait() as wpr:
                with out_acc_cb.reserve() as acc:
                    result = ttl.math.matmul(mav, wpr, acc)
                    acc.store(result)

            # Remaining chunks - accumulate
            for _ in range(NUM_MLP_CHUNKS - 1):
                with w_fc_chunk_cb.wait() as wfc, mlp_chunk_cb.reserve() as mh:
                    result = ttl.math.matmul(xv, wfc, mh)
                    mh.store(result)
                with mlp_chunk_cb.wait() as mhv, mlp_chunk_cb.reserve() as ma:
                    relu_x = ttl.math.relu(mhv)
                    ma.store(relu_x * relu_x)
                with mlp_chunk_cb.wait() as mav, w_proj_chunk_cb.wait() as wpr:
                    with partial_cb.reserve() as part:
                        result = ttl.math.matmul(mav, wpr, part)
                        part.store(result)
                with partial_cb.wait() as partv, out_acc_cb.wait() as acc:
                    with out_acc_cb.reserve() as new_acc:
                        new_acc.store(acc + partv)

            # Copy to output
            with out_acc_cb.wait() as acc, out_cb.reserve() as o:
                o.store(ttl.math.abs(acc))

    @ttl.datamovement()
    def dm_read():
        # Load x once
        with x_cb.reserve() as blk:
            tx = ttl.copy(x[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()

        # Stream weight chunks - MUST match compute loop count
        for chunk in range(NUM_MLP_CHUNKS):
            col_start = chunk * MLP_CHUNK_TILES
            col_end = col_start + MLP_CHUNK_TILES

            with w_fc_chunk_cb.reserve() as blk:
                tx = ttl.copy(w_fc[0:EMBD_TILES, col_start:col_end], blk)
                tx.wait()
            with w_proj_chunk_cb.reserve() as blk:
                tx = ttl.copy(w_proj[col_start:col_end, 0:EMBD_TILES], blk)
                tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:EMBD_TILES])
            tx.wait()
```

---

## Pattern 5: Softmax Decomposition

Softmax requires: max → shift → exp → sum → divide. Keep input in scope for two exp computations.

```python
@ttl.kernel(grid=(1, 1))
def softmax_kernel(x, scaler, out):
    """softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))"""
    SEQ_TILES = 4

    x_cb = ttl.make_circular_buffer_like(x, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)

    max_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    max_bcast_cb = ttl.make_circular_buffer_like(x, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    exp_cb = ttl.make_circular_buffer_like(x, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    sum_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    sum_bcast_cb = ttl.make_circular_buffer_like(x, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        # Keep x and scaler in scope for entire function
        with x_cb.wait() as xv, scaler_cb.wait() as sc:
            # Step 1: Find max
            with max_cb.reserve() as mx:
                max_val = ttl.math.reduce_max(xv, sc, mx, dims=[0, 1])
                mx.store(max_val)

            # Step 2: Broadcast max
            with max_cb.wait() as mxv, max_bcast_cb.reserve() as mxb:
                bcast = ttl.math.broadcast(mxv, mxb, dims=[0, 1])
                mxb.store(bcast)

            # Keep max_bcast in scope for both exp computations
            with max_bcast_cb.wait() as mxbv:
                # Step 3: Exp for sum
                with exp_cb.reserve() as ex:
                    shifted = xv - mxbv
                    ex.store(ttl.math.exp(shifted))

                # Step 4: Sum
                with exp_cb.wait() as exv, sum_cb.reserve() as sm:
                    sum_val = ttl.math.reduce_sum(exv, sc, sm, dims=[0, 1])
                    sm.store(sum_val)

                # Step 5: Broadcast sum
                with sum_cb.wait() as smv, sum_bcast_cb.reserve() as smb:
                    sum_bcast = ttl.math.broadcast(smv, smb, dims=[0, 1])
                    smb.store(sum_bcast)

                # Step 6: Final softmax = exp(x - max) / sum
                # xv and mxbv still in scope!
                with sum_bcast_cb.wait() as smbv, out_cb.reserve() as o:
                    shifted2 = xv - mxbv
                    exp_val2 = ttl.math.exp(shifted2)
                    o.store(exp_val2 / smbv)

    @ttl.datamovement()
    def dm_read():
        with x_cb.reserve() as blk:
            tx = ttl.copy(x[0:SEQ_TILES, 0:SEQ_TILES], blk)
            tx.wait()
        with scaler_cb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:SEQ_TILES])
            tx.wait()
```

---

## Pattern 6: RMSNorm with Reduce/Broadcast

```python
@ttl.kernel(grid=(1, 1))
def rmsnorm_kernel(x, weight, scaler, out):
    """RMSNorm: out = x * rsqrt(sum(x²)) * weight"""
    SEQ_TILES, EMBD_TILES = 4, 4

    x_cb = ttl.make_circular_buffer_like(x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2)
    weight_cb = ttl.make_circular_buffer_like(weight, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2)

    sq_cb = ttl.make_circular_buffer_like(x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2)
    reduce_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    bcast_cb = ttl.make_circular_buffer_like(x, shape=(SEQ_TILES, EMBD_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        with x_cb.wait() as xv, weight_cb.wait() as wv, scaler_cb.wait() as sc:
            # Square
            with sq_cb.reserve() as sq:
                sq.store(xv * xv)

            # Reduce to scalar
            with sq_cb.wait() as sqv, reduce_cb.reserve() as red:
                total = ttl.math.reduce_sum(sqv, sc, red, dims=[0, 1])
                red.store(total)

            # Rsqrt
            with reduce_cb.wait() as sumv, reduce_cb.reserve() as rsq:
                rsq.store(ttl.math.rsqrt(sumv))

            # Broadcast
            with reduce_cb.wait() as rsqv, bcast_cb.reserve() as bc:
                bc.store(ttl.math.broadcast(rsqv, bc, dims=[0, 1]))

            # Normalize: x * rsqrt * weight
            with bcast_cb.wait() as rsqrt_bcast, out_cb.reserve() as o:
                normalized = xv * rsqrt_bcast
                o.store(normalized * wv)

    @ttl.datamovement()
    def dm_read():
        with x_cb.reserve() as blk:
            tx = ttl.copy(x[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with weight_cb.reserve() as blk:
            tx = ttl.copy(weight[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with scaler_cb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:EMBD_TILES])
            tx.wait()
```
