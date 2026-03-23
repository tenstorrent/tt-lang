# tt-lang Element Access Spec: Fully On-Device Argmax

## Motivation

The Qwen decode argmax currently takes 3.8ms avg (1.2ms min) because the host must read back a tile range from the device to find the exact index of the maximum value. A fully on-device kernel would reduce this to ~0.5ms by scanning tile elements directly on the RISC-V datamovement cores.

## What's Needed

Three new capabilities in tt-lang's datamovement thread compiler:

### 1. `ttl.element_read(block, row, col)` → scalar

Read a single bf16/f32 element from a CB block at tile coordinates (row, col).

- **Thread**: DM only (NCRISC/BRISC)
- **Compiles to**: pointer arithmetic on L1 CB address + load
- **Tile layout**: Must handle face-based layout:
  - Face 0: rows 0-15, cols 0-15
  - Face 1: rows 0-15, cols 16-31
  - Face 2: rows 16-31, cols 0-15
  - Face 3: rows 16-31, cols 16-31

```python
@ttl.datamovement()
def write():
    with cb.wait() as blk:
        val = ttl.element_read(blk, 0, 5)  # read row 0, col 5
```

Generated C++ (bf16):
```cpp
uint16_t* base = (uint16_t*)get_read_ptr(cb_id);
// row=0, col=5 → face 0, local_row=0, local_col=5
uint16_t raw = base[0 * 16 + 5];
float val = bfloat16_to_float(raw);
```

### 2. `ttl.element_write(block, row, col, value)` → void

Write a single element to a CB block at tile coordinates (row, col).

```python
@ttl.datamovement()
def write():
    with cb.reserve() as blk:
        ttl.element_write(blk, 0, 0, 42)
```

### 3. Scalar variables + `if` in DM threads

DM threads need scalar integer/float variables that persist across loop iterations, plus `if` conditionals based on scalar comparisons.

```python
@ttl.datamovement()
def write():
    best_idx = 0xFFFFFFFF
    for t in range(chunk):
        with in_cb.wait() as blk:
            for c in range(32):
                val = ttl.element_read(blk, 0, c)
                if val == max_val:
                    best_idx = tile_start * 32 + c
```

**Note**: Basic scalar variables and loops may already work in tt-lang's DM thread. The main NEW primitives are `element_read`/`element_write`.

## Target Kernel: parallel_index_find

```python
@ttl.kernel(grid=(GRID_Y, GRID_X))
def parallel_index_find_kernel(logits, global_max, index_out):
    """Each core scans its tile range for the global max value.

    logits:     [32, V_padded]      — input logits
    global_max: [32, 32]            — broadcast global max (from kernel 2)
    index_out:  [32, N_CORES * 32]  — per-core found index
    """
    Nt = logits.shape[1] // TILE
    y_size, x_size = ttl.grid_size(dims=2)
    num_cores = y_size * x_size
    chunk = (Nt + num_cores - 1) // num_cores

    in_dfb = ttl.make_dataflow_buffer_like(logits, shape=(1, 1), buffer_factor=2)
    mx_dfb = ttl.make_dataflow_buffer_like(global_max, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(index_out, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        tile_start = nid * chunk

        with mx_dfb.reserve() as blk:
            tx = ttl.copy(global_max[0, 0], blk)
            tx.wait()

        for tid in range(chunk):
            col = (tile_start + tid) % Nt
            with in_dfb.reserve() as blk:
                tx = ttl.copy(logits[0, col], blk)
                tx.wait()

    @ttl.compute()
    def compute():
        # No compute needed — DM write thread handles scanning
        with mx_dfb.wait() as _:
            pass
        for _ in range(chunk):
            with in_dfb.wait() as _:
                pass

    @ttl.datamovement()
    def write():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        tile_start = nid * chunk

        # Read global max value (scalar)
        with mx_dfb.wait() as mx_blk:
            max_val = ttl.element_read(mx_blk, 0, 0)

        # Scan tiles for the max value
        best_idx = 0xFFFFFFFF
        for tid in range(chunk):
            col = (tile_start + tid) % Nt
            with in_dfb.wait() as blk:
                for c in range(32):
                    val = ttl.element_read(blk, 0, c)
                    if val == max_val:
                        global_col = col * 32 + c
                        if global_col < best_idx:
                            best_idx = global_col

        # Write result
        with out_dfb.reserve() as blk:
            ttl.element_write(blk, 0, 0, best_idx)
            tx = ttl.copy(blk, index_out[0, nid])
            tx.wait()
```

## Implementation Scope

| Component | File | Change |
|-----------|------|--------|
| New ops | `python/ttl/operators.py` | Add `element_read`, `element_write` syntax |
| Dialect | `include/ttlang/Dialect/TTL/IR/TTLOps.td` | Op definitions |
| Lowering | `lib/Dialect/TTL/Transforms/ConvertTTLToTTKernel.cpp` | Lower to L1 pointer ops |
| C++ emission | Existing EmitC pass | Emit pointer arithmetic |
| Test | `test/python/test_element_access.py` | Lit test |

## Expected Performance

With all three kernels on device:
- Kernel 1 (parallel_max_reduce): ~0.2ms
- Kernel 2 (global_max_reduce): ~0.05ms
- Kernel 3 (parallel_index_find): ~0.2ms (each core scans ~43 tiles × 32 cols = 1376 elements)
- Host readback (1 uint32): ~0.05ms
- **Total: ~0.5ms** (vs 3.8ms current, 11ms baseline)
