# Lowering Multi-tile Compute Operations

This document traces the lowering of a 2x2 multi-tile add operation from Python through the TTL pipeline to C++ kernel code. The reference test is [test/python/simple_add_multitile.py](https://github.com/tenstorrent/tt-lang/blob/main/test/python/simple_add_multitile.py).

## Python Input

```python
from ttlang import ttl, make_circular_buffer_like
from ttlang.ttl_api import Program
from ttlang.operators import copy

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

Pass: [lib/Dialect/TTL/Transforms/TTLTileAndAssignDST.cpp](https://github.com/tenstorrent/tt-lang/blob/main/lib/Dialect/TTL/Transforms/TTLTileAndAssignDST.cpp)

Inserts `ttl.copy_tile` to load tiles into DST registers. Assigns `dst_idx` attributes to tile math ops. The `ttl.linearized_index` computes tile position: for 2x2, maps (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3. Also computes peak DST usage per tile iteration and stores it as `ttl.dst_footprint` attribute on the compute op for use in dynamic DST index computation during lowering.

```mlir
%11 = ttl.compute ins(%4, %6 : ...) outs(%10 : ...) {...}
      {"ttl.dst_footprint" = 2 : i32} {
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

Inserts DST register lifecycle operations. `init_sfpu` initializes SFPU hardware. `tile_regs_acquire/release` bracket the entire compute+pack sequence. For multi-tile, `tile_regs_commit/wait` are placed outside the compute loop body to synchronize after all tiles are computed but before packing begins. `ttl.store` writes result tiles to output buffer.

```mlir
ttl.init_sfpu(%0, %2) : <[2, 2], !ttcore.tile<32x32, bf16>, 2>, <[2, 2], !ttcore.tile<32x32, bf16>, 2>
ttl.tile_regs_acquire

%11 = ttl.compute ins(...) outs(...) {...} {
^bb0(...):
  // ... copy_tile operations ...
  %15 = ttl.tile_add %dst_tile, %dst_tile_1 {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>
  ttl.yield %15 : !ttcore.tile<32x32, bf16>
} -> tensor<2x2x!ttcore.tile<32x32, bf16>>

// Sync placed outside compute loop for multi-tile
ttl.tile_regs_commit
ttl.tile_regs_wait

// Pack loop (separate from compute)
ttl.store %result, %output_cb : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>

ttl.tile_regs_release
```

### Stage 5: `ttl-lower-to-loops`

Pass: [lib/Dialect/TTL/Transforms/ConvertTTLComputeToSCF.cpp](https://github.com/tenstorrent/tt-lang/blob/main/lib/Dialect/TTL/Transforms/ConvertTTLComputeToSCF.cpp)

Materializes `ttl.compute` into two separate nested `scf.for` loop nests: one for compute operations and one for pack operations. Loop bounds come from tensor shape (2x2). `tensor.extract/insert` access individual tiles. `iter_args` carry output tensor between iterations (functional style). `affine.apply` evaluates the linearized index using loop induction variables. The `ttl.dst_footprint` attribute is propagated to the outermost loop of each nest for use in dynamic DST index computation.

```mlir
%c2 = arith.constant 2 : index
%c1 = arith.constant 1 : index
%c0 = arith.constant 0 : index

ttl.init_sfpu(%0, %2) : ...
ttl.tile_regs_acquire

// Compute loop - math operations only, no pack
scf.for %arg0 = %c0 to %c2 step %c1 {"ttl.dst_footprint" = 2 : i32} {
  scf.for %arg2 = %c0 to %c2 step %c1 {
    %extracted = tensor.extract %4[%arg0, %arg2] : tensor<2x2x!ttcore.tile<32x32, bf16>>
    %extracted_0 = tensor.extract %6[%arg0, %arg2] : tensor<2x2x!ttcore.tile<32x32, bf16>>

    %14 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg0, %arg2)

    %dst_token, %dst_tile = ttl.copy_tile %extracted, %14, %c0 : ...
    %dst_token_1, %dst_tile_2 = ttl.copy_tile %extracted_0, %14, %c1 : ...

    %16 = ttl.tile_add %dst_tile, %dst_tile_2 {dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>
  }
}

// Sync between compute and pack loops
ttl.tile_regs_commit
ttl.tile_regs_wait

// Pack loop - separate from compute
%11 = scf.for %arg0 = %c0 to %c2 step %c1 iter_args(%arg1 = %10) {"ttl.dst_footprint" = 2 : i32}
    -> (tensor<2x2x!ttcore.tile<32x32, bf16>>) {
  %13 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %arg1)
      -> (tensor<2x2x!ttcore.tile<32x32, bf16>>) {
    ttl.store %result, %output_cb : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    %inserted = tensor.insert %result into %arg3[%arg0, %arg2] : tensor<2x2x!ttcore.tile<32x32, bf16>>
    scf.yield %inserted : tensor<2x2x!ttcore.tile<32x32, bf16>>
  }
  scf.yield %13 : tensor<2x2x!ttcore.tile<32x32, bf16>>
}

ttl.tile_regs_release
```

### Stage 6: `convert-ttl-to-ttkernel`

Pass: [lib/Dialect/TTL/Transforms/ConvertTTLToTTKernel.cpp](https://github.com/tenstorrent/tt-lang/blob/main/lib/Dialect/TTL/Transforms/ConvertTTLToTTKernel.cpp)

Converts TTL operations to hardware-specific `ttkernel.*` operations. CB handles become typed `!ttkernel.cb`. Tile operations map to hardware intrinsics: `copy_tile_init/copy_tile`, `add_binary_tile_init/add_binary_tile`, `pack_tile`. DST indices are computed dynamically using affine maps: `base_dst_idx + footprint * linearize(ivs, ubs)` where footprint comes from the `ttl.dst_footprint` attribute on the enclosing loop.

```mlir
// Compute loop
scf.for %arg0 = %c0 to %c2 step %c1 {"ttl.dst_footprint" = 2 : i32} {
  scf.for %arg1 = %c0 to %c2 step %c1 {
    // Linearized tile index: i * 2 + j
    %lin_idx = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg0, %arg1)

    ttkernel.copy_tile_init(%0) : ...
    // Dynamic DST index: base(0) + footprint(2) * lin_idx
    %dst0 = affine.apply affine_map<(d0, d1) -> (d0 * 4 + d1 * 2)>(%arg0, %arg1)
    ttkernel.copy_tile(%0, %lin_idx, %dst0) : ...

    ttkernel.copy_tile_init(%1) : ...
    // Dynamic DST index: base(1) + footprint(2) * lin_idx
    %dst1 = affine.apply affine_map<(d0, d1) -> (d0 * 4 + d1 * 2 + 1)>(%arg0, %arg1)
    ttkernel.copy_tile(%1, %lin_idx, %dst1) : ...

    ttkernel.add_binary_tile_init() : ...
    ttkernel.add_binary_tile(%dst0, %dst1, %dst0) : ...
  }
}

ttkernel.tile_regs_commit() : ...
ttkernel.tile_regs_wait() : ...

// Pack loop
scf.for %arg0 = %c0 to %c2 step %c1 {"ttl.dst_footprint" = 2 : i32} {
  scf.for %arg1 = %c0 to %c2 step %c1 {
    %cb_idx = arith.muli %arg0, %c2 : index
    %cb_idx_1 = arith.addi %cb_idx, %arg1 : index
    // Dynamic DST index for pack: cbTileIndex * footprint
    %pack_dst = arith.muli %cb_idx_1, %c2 : index
    ttkernel.pack_tile(%pack_dst, %2, %cb_idx_1, false) : ...
  }
}
```

### Stage 7: `lower-affine`

Pass: MLIR standard pass (`mlir/Conversion/AffineToStandard`)

Lowers `affine.apply` to explicit arithmetic. The DST index maps like `affine_map<(d0, d1) -> (d0 * 4 + d1 * 2)>` become sequences of `arith.muli` and `arith.addi`.

```mlir
// Compute loop
scf.for %arg0 = %c0 to %c2 step %c1 {
  scf.for %arg1 = %c0 to %c2 step %c1 {
    // Linearized tile index: i * 2 + j
    %c2_0 = arith.constant 2 : index
    %lin_off = arith.muli %arg0, %c2_0 : index
    %lin_idx = arith.addi %lin_off, %arg1 : index

    ttkernel.copy_tile_init(%0) : ...
    // DST index for first operand: i * 4 + j * 2 (base=0, footprint=2, cols=2)
    %c4 = arith.constant 4 : index
    %dst0_row = arith.muli %arg0, %c4 : index
    %dst0_col = arith.muli %arg1, %c2_0 : index
    %dst0 = arith.addi %dst0_row, %dst0_col : index
    ttkernel.copy_tile(%0, %lin_idx, %dst0) : ...

    ttkernel.copy_tile_init(%1) : ...
    // DST index for second operand: i * 4 + j * 2 + 1 (base=1)
    %dst1_base = arith.addi %dst0_row, %dst0_col : index
    %c1_0 = arith.constant 1 : index
    %dst1 = arith.addi %dst1_base, %c1_0 : index
    ttkernel.copy_tile(%1, %lin_idx, %dst1) : ...

    ttkernel.add_binary_tile_init() : ...
    ttkernel.add_binary_tile(%dst0, %dst1, %dst0) : ...
  }
}

ttkernel.tile_regs_commit() : ...
ttkernel.tile_regs_wait() : ...

// Pack loop
scf.for %arg0 = %c0 to %c2 step %c1 {
  scf.for %arg1 = %c0 to %c2 step %c1 {
    %cb_off = arith.muli %arg0, %c2 : index
    %cb_idx = arith.addi %cb_off, %arg1 : index
    %pack_dst = arith.muli %cb_idx, %c2 : index
    ttkernel.pack_tile(%pack_dst, %2, %cb_idx, false) : ...
  }
}
```

## C++ Output

Three separate kernel files are generated: compute, data movement read, and data movement write.

### Compute Kernel

```cpp
namespace NAMESPACE {
void kernel_main() {
  int32_t v1 = 4;
  size_t v2 = 2;
  size_t v3 = 1;
  size_t v4 = 0;
  cb_wait_front(get_compile_time_arg_val(0), v1);
  cb_wait_front(get_compile_time_arg_val(1), v1);
  init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(2));
  tile_regs_acquire();

  // Compute loop
  for (size_t i5 = v4; i5 < v2; i5 += v3) {
    for (size_t j6 = v4; j6 < v2; j6 += v3) {
      // Linearized tile index: i * 2 + j
      size_t v7 = 2;
      size_t v8 = i5 * v7;
      size_t v9 = v8 + j6;

      copy_tile_init(get_compile_time_arg_val(0));
      // Dynamic DST index for first operand: i * 4 + j * 2
      size_t v10 = 4;
      size_t v11 = i5 * v10;
      size_t v12 = 2;
      size_t v13 = j6 * v12;
      size_t v14 = v11 + v13;
      copy_tile(get_compile_time_arg_val(0), v9, v14);

      copy_tile_init(get_compile_time_arg_val(1));
      // Dynamic DST index for second operand: i * 4 + j * 2 + 1
      size_t v15 = 4;
      size_t v16 = i5 * v15;
      size_t v17 = 2;
      size_t v18 = j6 * v17;
      size_t v19 = v16 + v18;
      size_t v20 = 1;
      size_t v21 = v19 + v20;
      copy_tile(get_compile_time_arg_val(1), v9, v21);

      add_binary_tile_init();
      add_binary_tile(v14, v21, v14);
    }
  }

  // Sync between compute and pack loops
  tile_regs_commit();
  tile_regs_wait();

  // Pack loop - separate from compute
  for (size_t i22 = v4; i22 < v2; i22 += v3) {
    for (size_t j23 = v4; j23 < v2; j23 += v3) {
      cb_reserve_back(get_compile_time_arg_val(2), v1);
      // CB tile index: i * 2 + j
      size_t v24 = i22 * v2;
      size_t v25 = v24 + j23;
      // Dynamic DST index for pack: cbTileIndex * footprint
      size_t v26 = v25 * v2;
      pack_tile<false>(v26, get_compile_time_arg_val(2), v25);
      cb_push_back(get_compile_time_arg_val(2), v1);
    }
  }

  tile_regs_release();
  return;
}
void MAIN { kernel_main(); }
}
```

## Generating This Documentation

Claude Code can generate pass-by-pass traces like this using the verbose MLIR output workflow. To trace an error or explore lowering for a new test:

```bash
TTLANG_VERBOSE_PASSES=1 python test/python/your_test.py 2>&1 > /tmp/pipeline.log
```

Then ask Claude to trace the pipeline output. `CLAUDE.md` tells claude how to extract relevant snippets from each pass transition and summarize the transformations. See `CLAUDE.md` "Workflow 1: Trace Issue Through Pass Pipeline" for details.
