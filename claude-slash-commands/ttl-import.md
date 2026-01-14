---
description: Import and translate a CUDA, Triton, PyTorch kernel, or TTNN program to TT-Lang DSL
argument-hint: <kernel-file-or-code>
---

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
from ttlang import ttl
import ttnn

@ttl.kernel(grid=(1, 1))  # (rows, cols) of Tensix cores
def my_kernel(input1, input2, output):
    # Create circular buffers - one per tensor
    # shape=(rows, cols) in tiles, buffer_factor for double-buffering
    input1_cb = ttl.make_circular_buffer_like(input1, shape=(1, 1), buffer_factor=2)
    input2_cb = ttl.make_circular_buffer_like(input2, shape=(1, 1), buffer_factor=2)
    output_cb = ttl.make_circular_buffer_like(output, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        # Wait for data from reader
        a = input1_cb.wait()
        b = input2_cb.wait()
        # Reserve output space
        o = output_cb.reserve()

        # Compute operations
        result = a + b

        # Store result and signal completion
        o.store(result)
        input1_cb.pop()
        input2_cb.pop()
        output_cb.push()

    @ttl.datamovement()
    def dm_read():
        # Reserve CB space, copy from DRAM, signal ready
        blk = input1_cb.reserve()
        tx = ttl.copy(input1[0, 0], blk)  # [row, col] in tiles
        tx.wait()
        input1_cb.push()

        blk2 = input2_cb.reserve()
        tx2 = ttl.copy(input2[0, 0], blk2)
        tx2.wait()
        input2_cb.push()

    @ttl.datamovement()
    def dm_write():
        # Wait for compute result, copy to DRAM
        blk = output_cb.wait()
        tx = ttl.copy(blk, output[0, 0])
        tx.wait()
        output_cb.pop()

    return ttl.Program(compute, dm_read, dm_write)(input1, input2, output)
```

### Using Context Managers (Recommended)

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

## Available Operations

### Binary Operations (on blocks)

```python
result = a + b      # Addition
result = a - b      # Subtraction
result = a * b      # Multiplication
result = a / b      # Division
result = a @ b      # Matrix multiplication
result = ttl.math.max(a, b)  # Element-wise maximum
```

### Unary Operations (ttl.math.*)

```python
result = ttl.math.exp(x)      # Exponential
result = ttl.math.log(x)      # Natural logarithm
result = ttl.math.sqrt(x)     # Square root
result = ttl.math.rsqrt(x)    # Reciprocal square root (1/sqrt(x))
result = ttl.math.tanh(x)     # Hyperbolic tangent
result = ttl.math.sigmoid(x)  # Sigmoid (1/(1+exp(-x)))
result = ttl.math.relu(x)     # ReLU (max(0, x))
result = ttl.math.abs(x)      # Absolute value
result = ttl.math.neg(x)      # Negation (-x)
result = ttl.math.sin(x)      # Sine
result = ttl.math.cos(x)      # Cosine
result = ttl.math.recip(x)    # Reciprocal (1/x)
```

### Operation Fusion

Operations chain automatically - no need for store/reload between ops:

```python
@ttl.compute()
def fused_compute():
    a = input_cb.wait()
    b = bias_cb.wait()
    o = output_cb.reserve()

    # All these ops fuse into one efficient compute body
    x = ttl.math.exp(a)
    y = x + b
    z = ttl.math.sigmoid(y)
    result = ttl.math.relu(z)

    o.store(result)
    # ...
```

The compiler fuses 20+ ops in a single compute function without issues.

## Multi-Tile Processing

For tensors larger than 32x32, process multiple tiles:

### Option 1: Large CB Shape (Single Core)

```python
# 64x64 tensor = 2x2 tiles
@ttl.kernel(grid=(1, 1))
def multitile_kernel(lhs, rhs, out):
    # CB holds all 4 tiles
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(2, 2), buffer_factor=2)
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape=(2, 2), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def compute():
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = l + r  # Operates on all tiles
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        blk = lhs_cb.reserve()
        tx = ttl.copy(lhs[0:2, 0:2], blk)  # Slice: rows 0-1, cols 0-1 (in tiles)
        tx.wait()
        lhs_cb.push()
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

    return ttl.Program(compute, dm_read, dm_write)(lhs, rhs, out)
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

## Translation Guide: GPU → TT-Lang

### Concept Mapping

| GPU Concept | TT-Lang Equivalent |
|------------|-------------------|
| Thread block / workgroup | Grid of Tensix cores (`grid=(rows, cols)`) |
| Shared memory | L1 via circular buffers |
| Global memory | DRAM with DMA transfers |
| Warp/wave operations | Tile-level operations (32x32) |
| `__syncthreads()` | CB `wait()`/`push()` synchronization |
| Kernel launch | `ttl.Program(...)()` execution |

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

    return ttl.Program(compute, dm_read, dm_write)(a, b, c)
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

**Example: Using TTNN for transpose**
```python
# If you need transpose and it's not in ttl.math:
input_transposed = ttnn.transpose(input_tensor, -2, -1)

# Then use TT-Lang kernel on the transposed input
my_kernel(input_transposed, output_tensor)
```

**Rule of thumb:**
1. Try to implement in TT-Lang first
2. If an op is missing, use TTNN for that specific op
3. The bulk of computation should be in TT-Lang for fusion benefits

## Common Patterns

### Softmax Decomposition

```python
@ttl.compute()
def softmax_compute():
    x = input_cb.wait()
    o = output_cb.reserve()

    # softmax(x) = exp(x) / sum(exp(x))
    exp_x = ttl.math.exp(x)
    # Note: reduce_sum requires additional setup - see tests
    # For now, use store/reload pattern or TTNN for reduction

    o.store(result)
```

### Fused Activation

```python
@ttl.compute()
def silu_compute():
    x = input_cb.wait()
    o = output_cb.reserve()

    # SiLU = x * sigmoid(x)
    result = x * ttl.math.sigmoid(x)

    o.store(result)
```

### Fused LayerNorm (partial)

```python
@ttl.compute()
def layernorm_elementwise():
    x = input_cb.wait()
    mean = mean_cb.wait()  # Pre-computed
    var = var_cb.wait()    # Pre-computed
    gamma = gamma_cb.wait()
    beta = beta_cb.wait()
    o = output_cb.reserve()

    # (x - mean) / sqrt(var + eps) * gamma + beta
    x_centered = x - mean
    x_norm = x_centered * ttl.math.rsqrt(var)  # Assumes eps already added
    result = x_norm * gamma + beta

    o.store(result)
```

## Debugging Tips

1. **Start simple**: Get a single-tile, single-op kernel working first
2. **Check shapes**: All dimensions must be multiples of 32
3. **Verify CB balance**: Every `wait()` needs `pop()`, every `reserve()` needs `push()`
4. **Use COMPILE_ONLY**: Set `TTLANG_COMPILE_ONLY=1` to check compilation without running
5. **Save MLIR**: Use `TTLANG_INITIAL_MLIR=/tmp/debug.mlir` to inspect generated IR

## Output

Provide:
1. The translated TT-Lang kernel with clear comments (save to file)
2. A test harness to verify correctness against the original
3. Notes on any TTNN ops used to fill gaps
4. Suggestions for further optimization if applicable
