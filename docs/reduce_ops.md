# Reduce Operations: `reduce_sum` and `reduce_max`

## Overview

`ttl.math.reduce_sum` and `ttl.math.reduce_max` reduce a tile along a specified dimension (rows or columns). They map directly to the hardware `reduce_init` / `reduce_tile` / `reduce_uninit` API and are available in both the simulator and compiler paths.

## API

```python
ttl.math.reduce_sum(input, scaler, output, dims=[0])
ttl.math.reduce_max(input, scaler, output, dims=[0])
```

| Argument | Description |
|----------|-------------|
| `input`  | Input tile block (CB-attached) |
| `scaler` | Scaler tile block (CB-attached). Each element is multiplied during reduction. Use all-ones for raw sum/max, or `1/N` for mean. |
| `output` | Output tile block (CB-attached, required for output CB tracking) |
| `dims`   | `[0]` reduces across rows (column-wise), `[1]` reduces across columns (row-wise) |

## Usage

```python
@ttl.kernel(grid=(1, 1))
def reduce_kernel(inp, scaler, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    sc_dfb  = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as i, sc_dfb.wait() as s, out_dfb.reserve() as o:
            o.store(ttl.math.reduce_sum(i, s, o, dims=[0]))

    @ttl.datamovement()
    def dm_read():
        blk = inp_dfb.reserve()
        tx = ttl.copy(inp[0, 0], blk); tx.wait(); blk.push()
        blk = sc_dfb.reserve()
        tx = ttl.copy(scaler[0, 0], blk); tx.wait(); blk.push()

    @ttl.datamovement()
    def dm_write():
        blk = out_dfb.wait()
        tx = ttl.copy(blk, out[0, 0]); tx.wait(); blk.pop()
```

### Computing mean with a scaler

```python
# scaler filled with 1/32 → output is column-wise mean
scaler_torch = torch.full((32, 32), 1.0 / 32.0, dtype=torch.bfloat16)
```

### Multi-tile

Multi-tile works the same way — each tile is reduced independently:

```python
inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 2), buffer_factor=2)
# ... same pattern, use inp[0:2, 0:2] in copy ...
```

## Output layout

The hardware always places reduce results in **row 0** of each output tile.

| `dims` | Operation | Row 0 contents |
|--------|-----------|----------------|
| `[0]`  | Column-wise (sum/max across rows) | `output[0, j] = f(input[:, j])` — all 32 values valid |
| `[1]`  | Row-wise (sum/max across columns) | `output[0, j] = f(input[j, :])` — first 16 values valid |

### Why only 16 values for `dims=[1]`?

A 32×32 tile is internally four 16×16 faces. `REDUCE_ROW` reduces each face independently and places results in column 0 of each face. After untilization:
- Face 0 column 0 → row 0, elements 0–15 (valid)
- Face 2 column 0 → rows 16–31, column 0 (not row 0 elements 16–31)

This is a hardware tile-format property, not a compiler bug. For `dims=[0]`, the packer mask configuration produces all 32 valid values in row 0.

## Lowering pipeline

```
Python DSL                  ttl.math.reduce_sum / reduce_max
    ↓ operators.py
TTL tensor ops              ttl.reduce_sum / ttl.reduce_max
    ↓ ConvertTTLToCompute
TTL tile ops                ttl.tile_reduce_sum / ttl.tile_reduce_max  (inside ttl.compute)
    ↓ ConvertTTLTileOpsToTTKernel
TTKernel ops                ttkernel.reduce_tile  (+ reduce_init / reduce_uninit via insert-inits pass)
    ↓ TTKernelToEmitC
C++ output                  reduce_init<PoolType, ReduceDim>() / reduce_tile<>() / reduce_uninit()
```
