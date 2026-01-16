# Lowering Multi-tile Compute Operations

This document traces the lowering of a 2x2 multi-tile add operation from Python through the TTL pipeline to C++ kernel code. The reference test is [test/python/simple_add_multitile.py](https://github.com/tenstorrent/tt-lang/blob/main/test/python/simple_add_multitile.py).

## Python Input

```python
from ttl import ttl, make_circular_buffer_like
from ttl.ttl_api import Program
from ttl.operators import copy

@ttl.kernel(grid=(1, 1))
def add_multitile_kernel(lhs, rhs, out):
    lhs_cb = make_circular_buffer_like(lhs, shape=(2, 2), buffer_factor=2)
    rhs_cb = make_circular_buffer_like(rhs, shape=(2, 2), buffer_factor=2)
    out_cb = make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def add_compute():
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = l + r
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    # ...data movement...

    return Program(add_compute, dm_read, dm_write)(lhs, rhs, out)
```

## Pass Pipeline

The TTL pipeline is defined in [lib/Dialect/TTL/Pipelines/TTLPipelines.cpp](https://github.com/tenstorrent/tt-lang/blob/main/lib/Dialect/TTL/Pipelines/TTLPipelines.cpp).

### Stage 1: Initial IR

High-level TTL operations on 2x2 tile tensors. The `ttl.add` operates on entire tensor, CB operations manage data flow.

```mlir
func.func @add_compute() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %0 = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %1 = ttl.bind_cb{cb_index = 1, buffer_factor = 2} : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %2 = ttl.bind_cb{cb_index = 2, buffer_factor = 2} : <[2, 2], !ttcore.tile<32x32, bf16>, 2>

  %3 = ttl.cb_wait %0 : ... -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %5 = ttl.cb_wait %1 : ... -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %7 = ttl.cb_reserve %2 : ... -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %9 = ttl.add %4, %6 : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>
                      -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  ttl.cb_pop %0 : ...
  ttl.cb_pop %1 : ...
  ttl.cb_push %2 : ...
  return
}
```

### Stage 2: `convert-ttl-to-compute`

Pass: [lib/Dialect/TTL/Transforms/ConvertTTLToCompute.cpp](https://github.com/tenstorrent/tt-lang/blob/main/lib/Dialect/TTL/Transforms/ConvertTTLToCompute.cpp)

Replaces tensor-level `ttl.add` with `ttl.compute` region containing element-wise `ttl.tile_add`. The `indexing_maps` define element-wise access patterns. Iteration is implicit in `ttl.compute` semantics.

```mlir
%9 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
%10 = ttl.attach_cb %9, %2 : ...

%11 = ttl.compute ins(%4, %6 : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>)
                  outs(%10 : tensor<2x2x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
^bb0(%arg0: !ttcore.tile<32x32, bf16>, %arg1: !ttcore.tile<32x32, bf16>, %arg2: !ttcore.tile<32x32, bf16>):
  %13 = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, bf16>
  ttl.yield %13 : !ttcore.tile<32x32, bf16>
} -> tensor<2x2x!ttcore.tile<32x32, bf16>>
```

### Stage 3: `ttl-tile-and-assign-dst`

Pass: [lib/Dialect/TTL/Transforms/TTLAssignDST.cpp](https://github.com/tenstorrent/tt-lang/blob/main/lib/Dialect/TTL/Transforms/TTLAssignDST.cpp)

Inserts `ttl.copy_tile` to load tiles into DST registers. Assigns `dst_idx` attributes to tile math ops. The `ttl.linearized_index` computes tile position: for 2x2, maps (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3.

```mlir
%11 = ttl.compute ins(%4, %6 : ...) outs(%10 : ...) {...} {
^bb0(%arg0: !ttcore.tile<32x32, bf16>, %arg1: !ttcore.tile<32x32, bf16>, %arg2: !ttcore.tile<32x32, bf16>):
  %13 = ttl.linearized_index affine_map<(d0, d1) -> (d0 * 2 + d1)> : index
  %c0 = arith.constant 0 : index
  %dst_token, %dst_tile = ttl.copy_tile %arg0, %13, %c0 : !ttcore.tile<32x32, bf16>, index, index
                                                        -> !ttl.dst, !ttcore.tile<32x32, bf16>

  %14 = ttl.linearized_index affine_map<(d0, d1) -> (d0 * 2 + d1)> : index
  %c1 = arith.constant 1 : index
  %dst_token_0, %dst_tile_1 = ttl.copy_tile %arg1, %14, %c1 : !ttcore.tile<32x32, bf16>, index, index
                                                            -> !ttl.dst, !ttcore.tile<32x32, bf16>

  %15 = ttl.tile_add %dst_tile, %dst_tile_1 {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>
  ttl.yield %15 : !ttcore.tile<32x32, bf16>
} -> tensor<2x2x!ttcore.tile<32x32, bf16>>
```

### Stage 4: `ttl-insert-tile-regs-sync`

Pass: [lib/Dialect/TTL/Transforms/TTLInsertTileRegsSync.cpp](https://github.com/tenstorrent/tt-lang/blob/main/lib/Dialect/TTL/Transforms/TTLInsertTileRegsSync.cpp)

Inserts DST register lifecycle operations. `init_sfpu` initializes SFPU hardware. `tile_regs_acquire/release` bracket compute. `tile_regs_commit/wait` synchronize the register pipeline inside the loop body. `ttl.store` writes result tiles to output buffer.

```mlir
ttl.init_sfpu(%0, %2) : <[2, 2], !ttcore.tile<32x32, bf16>, 2>, <[2, 2], !ttcore.tile<32x32, bf16>, 2>
ttl.tile_regs_acquire

%11 = ttl.compute ins(...) outs(...) {...} {
^bb0(...):
  // ... copy_tile operations ...
  %15 = ttl.tile_add %dst_tile, %dst_tile_1 {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>

  ttl.tile_regs_commit
  ttl.tile_regs_wait
  ttl.store %15, %7 : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>

  ttl.yield %15 : !ttcore.tile<32x32, bf16>
} -> tensor<2x2x!ttcore.tile<32x32, bf16>>

ttl.tile_regs_release
```

### Stage 5: `ttl-lower-to-loops`

Pass: [lib/Dialect/TTL/Transforms/ConvertTTLComputeToSCF.cpp](https://github.com/tenstorrent/tt-lang/blob/main/lib/Dialect/TTL/Transforms/ConvertTTLComputeToSCF.cpp)

Materializes `ttl.compute` into explicit nested `scf.for` loops. Loop bounds come from tensor shape (2x2). `tensor.extract/insert` access individual tiles. `iter_args` carry output tensor between iterations (functional style). `affine.apply` evaluates the linearized index using loop induction variables.

```mlir
%c2 = arith.constant 2 : index
%c1 = arith.constant 1 : index
%c0 = arith.constant 0 : index

ttl.init_sfpu(%0, %2) : ...
ttl.tile_regs_acquire

%11 = scf.for %arg0 = %c0 to %c2 step %c1 iter_args(%arg1 = %10) -> (tensor<2x2x!ttcore.tile<32x32, bf16>>) {
  %13 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %arg1) -> (tensor<2x2x!ttcore.tile<32x32, bf16>>) {

    %extracted = tensor.extract %4[%arg0, %arg2] : tensor<2x2x!ttcore.tile<32x32, bf16>>
    %extracted_0 = tensor.extract %6[%arg0, %arg2] : tensor<2x2x!ttcore.tile<32x32, bf16>>

    %14 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg0, %arg2)
    %15 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg0, %arg2)

    %dst_token, %dst_tile = ttl.copy_tile %extracted, %14, %c0 : ...
    %dst_token_1, %dst_tile_2 = ttl.copy_tile %extracted_0, %15, %c1 : ...

    %16 = ttl.tile_add %dst_tile, %dst_tile_2 {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>

    ttl.tile_regs_commit
    ttl.tile_regs_wait
    ttl.store %16, %7 : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>

    %inserted = tensor.insert %16 into %arg3[%arg0, %arg2] : tensor<2x2x!ttcore.tile<32x32, bf16>>
    scf.yield %inserted : tensor<2x2x!ttcore.tile<32x32, bf16>>
  }
  scf.yield %13 : tensor<2x2x!ttcore.tile<32x32, bf16>>
}

ttl.tile_regs_release
```

### Stage 6: `convert-ttl-to-ttkernel`

Pass: [lib/Dialect/TTL/Transforms/ConvertTTLToTTKernel.cpp](https://github.com/tenstorrent/tt-lang/blob/main/lib/Dialect/TTL/Transforms/ConvertTTLToTTKernel.cpp)

Converts TTL operations to hardware-specific `ttkernel.*` operations. CB handles become typed `!ttkernel.cb`. Tile operations map to hardware intrinsics: `copy_tile_init/copy_tile`, `add_binary_tile_init/add_binary_tile`, `pack_tile`.

```mlir
scf.for %arg0 = %c0 to %c2 step %c1 {
  scf.for %arg1 = %c0 to %c2 step %c1 {

    %14 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg0, %arg1)

    ttkernel.copy_tile_init(%0) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%0, %14, %c0) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    ttkernel.copy_tile_init(%1) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%1, %14, %c1) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index, index) -> ()

    ttkernel.add_binary_tile_init() : () -> ()
    ttkernel.add_binary_tile(%c0, %c1, %c0) : (index, index, index) -> ()

    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %2, %c0, false) : (index, !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index) -> ()
  }
}
```

### Stage 7: `lower-affine`

Pass: MLIR standard pass (`mlir/Conversion/AffineToStandard`)

Lowers `affine.apply` to explicit arithmetic. The linearized index `affine_map<(d0, d1) -> (d0 * 2 + d1)>` becomes `arith.muli` and `arith.addi`.

```mlir
scf.for %arg0 = %c0 to %c2 step %c1 {
  scf.for %arg1 = %c0 to %c2 step %c1 {

    %c2_0 = arith.constant 2 : index
    %3 = arith.muli %arg0, %c2_0 overflow<nsw> : index   // row * 2
    %4 = arith.addi %3, %arg1 : index                    // + col

    ttkernel.copy_tile_init(%0) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%0, %4, %c0) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    ttkernel.copy_tile_init(%1) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%1, %4, %c1) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index, index) -> ()

    ttkernel.add_binary_tile_init() : () -> ()
    ttkernel.add_binary_tile(%c0, %c1, %c0) : (index, index, index) -> ()

    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %2, %c0, false) : (index, !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index) -> ()
  }
}
```

## C++ Output

Three separate kernel files are generated: compute, data movement read, and data movement write.

### Compute Kernel

```cpp
void kernel_main() {
  constexpr int32_t num_tiles = 4;
  constexpr size_t tile_rows = 2;
  constexpr size_t tile_cols = 2;

  cb_wait_front(get_compile_time_arg_val(0), num_tiles);
  cb_wait_front(get_compile_time_arg_val(1), num_tiles);
  cb_reserve_back(get_compile_time_arg_val(2), num_tiles);

  init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(2));
  tile_regs_acquire();

  for (size_t row = 0; row < tile_rows; row++) {
    for (size_t col = 0; col < tile_cols; col++) {
      size_t linear_idx = row * tile_cols + col;

      copy_tile_init(get_compile_time_arg_val(0));
      copy_tile(get_compile_time_arg_val(0), linear_idx, 0);
      copy_tile_init(get_compile_time_arg_val(1));
      copy_tile(get_compile_time_arg_val(1), linear_idx, 1);

      add_binary_tile_init();
      add_binary_tile(0, 1, 0);

      tile_regs_commit();
      tile_regs_wait();
      pack_tile<false>(0, get_compile_time_arg_val(2), 0);
    }
  }

  tile_regs_release();
  cb_pop_front(get_compile_time_arg_val(0), num_tiles);
  cb_pop_front(get_compile_time_arg_val(1), num_tiles);
  cb_push_back(get_compile_time_arg_val(2), num_tiles);
}
```

## Generating This Documentation

Claude Code can generate pass-by-pass traces like this using the verbose MLIR output workflow. To trace an error or explore lowering for a new test:

```bash
TTLANG_VERBOSE_PASSES=1 python test/python/your_test.py 2>&1 > /tmp/pipeline.log
```

Then ask Claude to trace the pipeline output. `CLAUDE.md` tells claude how to extract relevant snippets from each pass transition and summarize the transformations. See `CLAUDE.md` "Workflow 1: Trace Issue Through Pass Pipeline" for details.
