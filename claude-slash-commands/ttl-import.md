---
description: Import and translate a CUDA, Triton, PyTorch kernel, or TTNN program to TT-Lang DSL
argument-hint: <kernel-file-or-code>
---

## Tools Available

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

### Binary Operators

```python
result = a + b      # Element-wise addition
result = a - b      # Element-wise subtraction
result = a * b      # Element-wise multiplication
result = a / b      # Element-wise division
# NOTE: a @ b does NOT work! Use ttl.matmul() instead (see below)
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
# ttl.matmul is ACCUMULATING: result = a @ b + c
# The third argument 'c' is both input (accumulator) AND output
result = ttl.matmul(a, b, c)  # c += a @ b, returns updated c

# Example usage:
o = out_cb.reserve()
result = ttl.matmul(a_tile, b_tile, o)  # o is the accumulator
o.store(result)
```

### Power (scalar integer exponent)

```python
# Raises each element to an integer power
result = ttl.power(x, 2)  # x^2
result = ttl.power(x, 3)  # x^3
```

### Transpose (32x32 tile only)

```python
# Transposes a single 32x32 tile
# NOTE: Takes output block as second argument
o = out_cb.reserve()
result = ttl.transpose(input_tile, o)
o.store(result)
```

### Reductions (require scaler tensor)

```python
# Reductions need a "scaler" tensor (typically all 1.0s)
# dim options: "scalar" (default), "row", "col"
# NOTE: Takes output block as third argument

scaler = scaler_cb.wait()  # Tensor of 1.0s
o = out_cb.reserve()

# Scalar reduction (sum/max of entire tile -> result in [0,0])
result = ttl.reduce_sum(input_tile, scaler, o)
result = ttl.reduce_max(input_tile, scaler, o)

# Row reduction (sum/max each row -> result in column 0)
result = ttl.reduce_sum(input_tile, scaler, o, dim="row")

# Column reduction (sum/max each column -> result in row 0)
result = ttl.reduce_sum(input_tile, scaler, o, dim="col")

o.store(result)
```

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

No `softmax` op? Decompose it:
```python
# softmax(x) = exp(x) / sum(exp(x))
exp_x = ttl.math.exp(x)
sum_exp = ttl.reduce_sum(exp_x, scaler, o, dim="scalar")  # Needs scaler tensor of 1.0s
inv_sum = ttl.math.recip(sum_exp)  # 1/sum
result = exp_x * inv_sum           # Or use division: exp_x / sum_exp
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
    scaler = scaler_cb.wait()  # Need a tensor of 1.0s for reduce
    o = output_cb.reserve()

    # softmax(x) = exp(x) / sum(exp(x))
    exp_x = ttl.math.exp(x)

    # Reduce needs scaler and output block
    sum_exp = ttl.reduce_sum(exp_x, scaler, o, dim="scalar")
    # Note: result is in [0,0] only - need broadcast for element-wise ops

    # Option 1: Use recip + multiply
    inv_sum = ttl.math.recip(sum_exp)
    result = exp_x * inv_sum

    # Option 2: Use division directly
    # result = exp_x / sum_exp

    o.store(result)
    # ... pop/push ...
```

### Broadcasting a Scalar to a Tile

After `reduce_sum(..., dim="scalar")`, only position [0,0] has the valid result.
To use it in element-wise ops, you need to broadcast it to all positions.

**Option 1: Pre-create a broadcast tensor (simplest)**
```python
# Before the kernel, create a tensor where all elements equal the scalar
# Use TTNN or torch to fill a 32x32 tensor with the value
broadcast_tensor = ttnn.full((32, 32), scalar_value, ...)
```

**Option 2: Use datamovement to replicate**
```python
@ttl.datamovement()
def broadcast_scalar():
    # Read the single-value tile
    src_blk = scalar_cb.wait()

    # Write to multiple positions in output (loop over tile positions)
    for row in range(32):
        for col in range(32):
            # Copy [0,0] value to [row, col] in destination
            # This requires element-level access - may need TTNN
            pass

    scalar_cb.pop()
```

**Option 3: Use TTNN for broadcast (recommended for complex cases)**
```python
# After your kernel produces the scalar result:
scalar_tile = ttnn.to_torch(result_tensor)
scalar_value = scalar_tile[0, 0].item()
broadcast_tensor = ttnn.full((32, 32), scalar_value, device=device, ...)
# Then use broadcast_tensor in next operation
```

**Note:** For row/column reductions (`dim="row"` or `dim="col"`), the result is already
partially broadcast along one axis, which may be sufficient for some algorithms.

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
