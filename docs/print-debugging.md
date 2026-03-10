# TTL Debug Print Spec

## Overview

Python `print()` inside TTL kernels lowers to device debug prints via a `ttl.dprint` op and a dedicated lowering pass. The pass has full compiler context (DST assignments, CB indices, thread types) and emits the appropriate tt-metal DPRINT calls.

Enabled at runtime by `TT_METAL_DPRINT_CORES=0,0`. Zero overhead when not set (tt-metal compiles DPRINT to dead code).

Note: prints can be extremely large and slow! It is recommended to save to a file that you can grep, modify your program to only run on small targetted inputs, and only place prints in targetted temporary places.

Note: prints in compute will overlap and cause confusing output unless guarded with thread=X.

## Python API

### Scalars

```python
print("hello world")
print("x =", x, "y =", y)
print(42)
print(3.14)
```

Supported argument types: string constants, integer constants, float constants, integer variables (index, i32). Error on unsupported types.

### Circular buffer details

```python
with inp_dfb.wait() as tile:
    print(inp_dfb)
```

Prints CB metadata: size, limit, page_size, num_pages, rd_ptr, wr_ptr.

### Tile from CB (full tile)

```python
with inp_dfb.wait() as tile:
    print(tile, thread="pack")
```

Prints the full 32x32 tile contents from the CB. The tile must be live (between wait/pop or reserve/push).

Note: will dump all registers in a block if using multi-tile block size (cb shape > 1x1) as the print will be inside the loop generated.

Note: unsupported on math thread.

Example output:
```
0:(x=1,y=1):TR2: ======
0:(x=1,y=1):TR2: 0 : 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0:(x=1,y=1):TR2: 1 : 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
...
0:(x=1,y=1):TR2: 31 : 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0:(x=1,y=1):TR2: ++++++
0:(x=1,y=1):TR2: ======
```

### Tensor pages

```python
print(inp, num_pages=2)
```

Prints raw memory pages from a tensor's backing buffer. Data format (bf16, f32) and L1 address derived from the tensor type and layout during lowering. `num_pages` defaults to 1.

### DST register dump

```python
result = ttl.exp(i)
print(_dump_dst_registers=True, label="after exp")
```

Dumps all DST register slots that are live at this program point. The pass resolves assigned DST indices and includes the label and producing op name for each slot.

Note: will dump all registers in a block if using multi-tile block size (cb shape > 1x1) as the print will be inside the loop generated.

Note: only supports bf16.

### Thread conditioning

Any print can be conditioned on a specific thread:

```python
print(tile, thread="math")
print(tile, thread="pack")
print(inp_dfb, thread="unpack")
```

When `thread` is specified, the print is wrapped in the corresponding `DPRINT_MATH(...)`, `DPRINT_PACK(...)`, or `DPRINT_UNPACK(...)` macro. In compute kernels, the thread is automatically selected based on the print mode when no explicit `thread` is given: scalar and DST prints use `math`, CB and tile prints use `pack`. Tensor page prints (`num_pages=`) are only supported in datamovement kernels. In datamovement kernels, no wrapping is applied when `thread` is omitted.

## In depth + code gen

### Scalar

```python
print("x =", x)
```
```cpp
ttmlir::dprint("x = ", v0, "\n");
```

Scalars can stay on the existing `ttkernel.dprint` -> `ttmlir::dprint` path. The lowering pass does not need to touch these.

### CB details

```python
print(inp_dfb)
```
```cpp
print_cb_details(get_compile_time_arg_val(0));
```

Pass resolves the CB index from the `cb_index` attribute on the defining `bind_cb` (or the lowered ttkernel compile-time arg).

### Tile (full)

```python
with inp_dfb.wait() as tile:
    print(tile)
```
```cpp
print_full_tile(get_compile_time_arg_val(0), 0, true);
```

Pass traces the tile value back to its CB via `cb_wait`/`attach_cb`, resolves the CB index and tile index within the block.

### Tensor pages

```python
print(inp, num_pages=2)
```
```cpp
// bf16 tensor example
print_bf16_pages(get_read_ptr(get_compile_time_arg_val(0)), 1024, 2);
```

Pass derives the data format from the tensor element type (`bf16` -> `print_bf16_pages`, `f32` -> `print_f32_pages`) and the L1 address from the tensor accessor. `num_pages` comes from the op attribute.

Should only be used in datamovement threads.

### DST dump

```python
result = ttl.exp(i)
print(_dump_dst_registers=True, label="after exp")
```
```cpp
{
  DPRINT << "=== after exp ===" << ENDL();
  DPRINT << "DST[0] (ttl.exp):" << ENDL();
  dprint_tensix_dest_reg(0);
}
```

Pass walks backward from the op to find all tile values with DST slot assignments at this program point. For each live slot, emits `dprint_tensix_dest_reg(slot)` with a label identifying the producing op.

### Thread conditioning

```python
print(tile, thread="math")
```
```cpp
DPRINT_MATH(
  print_full_tile(get_compile_time_arg_val(0), 0, true);
);
```

Wraps the entire emitted block in the specified thread macro. In compute kernels, the thread is auto-selected per mode when not specified (scalar/DST -> math, CB/tile -> pack).

## Example: instrumented compute kernel

```python
@ttl.compute()
def compute():
    with inp_dfb.wait() as i, out_dfb.reserve() as o:
        print("compute start")
        print(inp_dfb)
        result = ttl.exp(i)
        print(_dump_dst_registers=True, label="after exp")
        o.store(result)
```

Generated C++ (compute kernel):

```cpp
void kernel_main() {
  // ...
  cb_wait_front(get_compile_time_arg_val(0), 1);
  cb_reserve_back(get_compile_time_arg_val(1), 1);

  ttmlir::dprint("compute start\n");
  print_cb_details(get_compile_time_arg_val(0));

  tile_regs_acquire();
  copy_tile_init(get_compile_time_arg_val(0));
  copy_tile(get_compile_time_arg_val(0), 0, 0);
  exp_tile_init();
  exp_tile(0);

  {
    DPRINT << "=== after exp ===" << ENDL();
    DPRINT << "DST[0] (ttl.exp):" << ENDL();
    dprint_tensix_dest_reg(0);
  }

  tile_regs_commit();
  tile_regs_wait();
  pack_tile<true>(0, get_compile_time_arg_val(1), 0);
  tile_regs_release();

  cb_pop_front(get_compile_time_arg_val(0), 1);
  cb_push_back(get_compile_time_arg_val(1), 1);
}
```
