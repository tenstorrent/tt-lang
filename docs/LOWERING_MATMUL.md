# Lowering Fused Matmul with Accumulation

This document traces the lowering of `relu(A @ B + C)` — a block matrix multiply-accumulate with a post-matmul activation — through the TTL pipeline from Python to C++ kernel code. The operands are `A[2x1]`, `B[1x2]`, and `C[2x2]` (in `bf16` 32x32 tiles), producing a `2x2` output (4 tiles).

Three operations fuse into a single compute region:
- **Accumulation**: `matmul_block` inherently performs `DST += A*B`, so pre-loading `C` via `copy_tile` produces `C + A*B` without a separate add instruction.
- **Activation**: relu operates in-place on the DST registers after the matmul, before packing.
- **Block-level matmul**: a single `matmul_block(rt=2, ct=2)` call processes all 4 output tiles.

Reference test: `test_matmul_add_relu` in [test/python/test_matmul_acc.py](../test/python/test_matmul_acc.py).

## Python Input

```python
@ttl.kernel(grid=(1, 1))
def matmul_add_relu_kernel(a, b, c, out):
    Mt = a.shape[0] // TILE   # 2
    Nt = b.shape[1] // TILE   # 2

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(Mt, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, Nt), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(Mt, Nt), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), buffer_factor=2)

    @ttl.compute()
    def mm_compute():
        with (
            a_dfb.wait() as a_blk,
            b_dfb.wait() as b_blk,
            c_dfb.wait() as c_blk,
        ):
            with out_dfb.reserve() as o:
                o.store(ttl.math.relu(a_blk @ b_blk + c_blk))

    # ...data movement...
```

## Pass Pipeline

The TTL pipeline is defined in [lib/Dialect/TTL/Pipelines/TTLPipelines.cpp](../lib/Dialect/TTL/Pipelines/TTLPipelines.cpp). The matmul fusion path goes through `convert-ttl-to-compute` (fusion), `ttl-assign-dst`, and `ttl-lower-matmul-block` (block expansion). The matmul compute bypasses `ttl-insert-tile-regs-sync` and `ttl-lower-to-loops`; sync and tile expansion are handled by `lower-matmul-block`.

### Stage 1: Initial IR

Tensor-level TTL operations. `ttl.matmul` multiplies `A[2x1]` by `B[1x2]`, `ttl.add` adds the bias `C[2x2]`, and `ttl.relu` is applied elementwise before `ttl.store`.

```mlir
%mm = ttl.matmul %a, %b : tensor<2x1x!ttcore.tile<32x32, bf16>>,
                           tensor<1x2x!ttcore.tile<32x32, bf16>>
                        -> tensor<2x2x!ttcore.tile<32x32, bf16>>
%sum = ttl.add %mm, %c  : tensor<2x2x...>, tensor<2x2x...> -> tensor<2x2x...>
%act = ttl.relu %sum     : tensor<2x2x...> -> tensor<2x2x...>
ttl.store %act, %reserve : tensor<2x2x...>, tensor<2x2x...>
```

### Stage 2: `convert-ttl-to-compute`

Pass: [lib/Dialect/TTL/Transforms/ConvertTTLToCompute.cpp](../lib/Dialect/TTL/Transforms/ConvertTTLToCompute.cpp)

The `convert-ttl-to-compute` pass lowers tensor-level ops into `ttl.compute` regions containing tile-level ops. When a fusable op's input is not CB-attached, `traceFusionToRoots` ([lib/Dialect/TTL/IR/TTLOpsUtils.cpp](../lib/Dialect/TTL/IR/TTLOpsUtils.cpp)) walks the SSA chain to find CB-attached roots and fusable intermediate ops. Here it traces from the store's input through `relu -> add -> matmul`. The matmul is a fusable leaf: both CB-attached inputs become roots. The add is folded into a 3-operand `tile_matmul_block(a, b, accumulator)` — no `tile_add` is emitted. The relu is emitted in-place after the matmul. All four tensor-level ops collapse into one `ttl.compute`.

The indexing maps encode the broadcast pattern: `A[2x1]` broadcasts along `d1` (`(d0,d1) -> (d0,0)`), `B[1x2]` along `d0` (`(d0,d1) -> (0,d1)`), `C[2x2]` and output use identity maps.

```mlir
%result = ttl.compute
    ins(%a, %b, %c : tensor<2x1x...>, tensor<1x2x...>, tensor<2x2x...>)
    outs(%init : tensor<2x2x...>)
    {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>,   // A: broadcast d1
                      affine_map<(d0, d1) -> (0, d1)>,   // B: broadcast d0
                      affine_map<(d0, d1) -> (d0, d1)>,  // C: identity
                      affine_map<(d0, d1) -> (d0, d1)>], // out: identity
     iterator_types = ["parallel", "parallel"]} {
^bb0(%a_t: tile, %b_t: tile, %c_t: tile, %out_t: tile):
  %mm = ttl.tile_matmul_block %a_t, %b_t, %c_t : tile, tile, tile -> tile
  %r = ttl.tile_relu %mm : tile
  ttl.tile_store %r, %view[%i, %j] : tile, tensor<2x2x...>
  ttl.yield
}
```

### Stage 3: `ttl-assign-dst`

Pass: [lib/Dialect/TTL/Transforms/TTLAssignDST.cpp](../lib/Dialect/TTL/Transforms/TTLAssignDST.cpp)

Assigns DST register indices. The matmul's accumulator operand and output are merged to the same DST slot (both must occupy `DST[0]` since `matmul_block` accumulates in-place). The relu shares `DST[0]` via `InPlaceOpTrait`. The `unroll_factor = 4` indicates the `2x2` output (4 tiles) fits in `bf16` DST capacity (8 tiles).

```mlir
^bb0(%a_t: tile, %b_t: tile, %c_t: tile, %out_t: tile):
  %mm = ttl.tile_matmul_block %a_t, %b_t, %c_t {dst_idx = 0 : i32} : ...
  %r = ttl.tile_relu %mm {dst_idx = 0 : i32} : tile
  ttl.tile_store %r, %view[%i, %j] : ...
  ttl.yield
```

### Stage 4: `ttl-lower-matmul-block`

Pass: [lib/Dialect/TTL/Transforms/TTLLowerMatmulBlock.cpp](../lib/Dialect/TTL/Transforms/TTLLowerMatmulBlock.cpp)

Replaces the `ttl.compute` region with a linear sequence of tile-level ops. The matmul block dimensions (`rt=2`, `ct=2`) are derived from the operand tensor shapes. The accumulator tensor is passed as the 3rd operand of `tile_matmul_block`; TTKernel lowering emits `rt*ct` `copy_tile` ops from it. Per-tile unary ops and stores are expanded to `M*N` copies with distinct `dst_idx` values.

`insert-tile-regs-sync` skips matmul computes (detected via `containsOp<TileMatmulBlockOp>`); sync ops are emitted here instead.

```mlir
ttl.tile_regs_acquire

// Single matmul_block call — accumulator %c passed as 3rd operand.
%mm = ttl.tile_matmul_block %a, %b, %c {dst_idx = 0 : i32}
    : tensor<2x1x...>, tensor<1x2x...>, tensor<2x2x...> -> tile

// Per-tile relu expansion (M*N = 4 copies).
ttl.tile_relu %placeholder {dst_idx = 0 : i32} : tile
ttl.tile_relu %placeholder {dst_idx = 1 : i32} : tile
ttl.tile_relu %placeholder {dst_idx = 2 : i32} : tile
ttl.tile_relu %placeholder {dst_idx = 3 : i32} : tile

ttl.tile_regs_commit
ttl.tile_regs_wait

// Per-tile store expansion (M*N = 4 copies).
ttl.tile_store %placeholder, %view[0, 0] {dst_idx = 0 : i32} : ...
ttl.tile_store %placeholder, %view[0, 1] {dst_idx = 1 : i32} : ...
ttl.tile_store %placeholder, %view[1, 0] {dst_idx = 2 : i32} : ...
ttl.tile_store %placeholder, %view[1, 1] {dst_idx = 3 : i32} : ...

ttl.tile_regs_release
```

### Stage 5: `convert-ttl-to-ttkernel` + `ttkernel-insert-inits`

Pass: [lib/Dialect/TTL/Transforms/ConvertTTLToTTKernel.cpp](../lib/Dialect/TTL/Transforms/ConvertTTLToTTKernel.cpp)

TTL ops are converted 1:1 to TTKernel hardware ops. The 3-operand `tile_matmul_block` emits `rt*ct` `copy_tile` ops for the accumulator CB before the `experimental::matmul_block` call. Init ops are inserted by `ttkernel-insert-inits`: `mm_block_init` (full init) before the sync region, `copy_tile_init` before the first copy, `mm_block_init_short` before the matmul, `relu_tile_init` before the first relu.

```mlir
"ttkernel.mm_block_init"(%a_cb, %b_cb, %out_cb, ...)

ttkernel.tile_regs_acquire

"ttkernel.copy_tile_init"(%c_cb)
ttkernel.copy_tile(%c_cb, %c0, %c0)       // C[0,0] -> DST[0]
ttkernel.copy_tile(%c_cb, %c1, %c1)       // C[0,1] -> DST[1]
ttkernel.copy_tile(%c_cb, %c2, %c2)       // C[1,0] -> DST[2]
ttkernel.copy_tile(%c_cb, %c3, %c3)       // C[1,1] -> DST[3]

"ttkernel.mm_block_init_short"(%a_cb, %b_cb, ...)
"ttkernel.experimental::matmul_block"(%a_cb, %b_cb, %c0, %c0, %c0,
    /*transpose=*/0, /*ct=*/2, /*rt=*/2, /*kt=*/1, /*nt=*/2)
                                            // DST[0..3] += A[2x1] * B[1x2]

"ttkernel.relu_tile_init"()
ttkernel.relu_tile(%c1)                    // relu(DST[1])
ttkernel.relu_tile(%c2)                    // relu(DST[2])
ttkernel.relu_tile(%c3)                    // relu(DST[3])
ttkernel.relu_tile(%c0)                    // relu(DST[0])

ttkernel.tile_regs_commit
ttkernel.tile_regs_wait

ttkernel.pack_tile(%c0, %out_cb, %c0)     // DST[0] -> out[0,0]
ttkernel.pack_tile(%c1, %out_cb, %c1)     // DST[1] -> out[0,1]
ttkernel.pack_tile(%c2, %out_cb, %c2)     // DST[2] -> out[1,0]
ttkernel.pack_tile(%c3, %out_cb, %c3)     // DST[3] -> out[1,1]

ttkernel.tile_regs_release
```

## C++ Output

### Compute Kernel

```cpp
void kernel_main() {
  // CB handles: a=CTA[0], b=CTA[1], c=CTA[2], out=CTA[3]
  // Constants: v2=2 (rt, ct), v1=1 (kt)

  cb_wait_front(get_compile_time_arg_val(0), 2);   // A: 2 tiles
  cb_wait_front(get_compile_time_arg_val(1), 2);   // B: 2 tiles
  cb_wait_front(get_compile_time_arg_val(2), 4);   // C: 4 tiles
  cb_reserve_back(get_compile_time_arg_val(3), 4);  // out: 4 tiles

  mm_block_init(a_cb, b_cb, out_cb, /*transpose=*/0,
                /*ct=*/2, /*rt=*/2, /*kt=*/1);

  tile_regs_acquire();

  // Phase 1: Pre-load bias C into DST[0..3]
  copy_tile_init(c_cb);
  copy_tile(c_cb, 0, 0);                  // C[0,0] -> DST[0]
  copy_tile(c_cb, 1, 1);                  // C[0,1] -> DST[1]
  copy_tile(c_cb, 2, 2);                  // C[1,0] -> DST[2]
  copy_tile(c_cb, 3, 3);                  // C[1,1] -> DST[3]

  // Phase 2: Matmul accumulates (DST += A * B)
  mm_block_init_short(a_cb, b_cb, /*transpose=*/0,
                      /*ct=*/2, /*rt=*/2, /*kt=*/1);
  experimental::matmul_block(a_cb, b_cb, 0, 0, 0,
      /*transpose=*/0, /*ct=*/2, /*rt=*/2, /*kt=*/1, /*nt=*/2);

  // Phase 3: Relu in-place on each DST tile
  relu_tile_init();
  relu_tile(1);
  relu_tile(2);
  relu_tile(3);
  relu_tile(0);

  // Phase 4: Pack all 4 tiles to output CB
  tile_regs_commit();
  tile_regs_wait();
  pack_tile<true>(0, out_cb, 0);           // DST[0] -> out[0,0]
  pack_tile<true>(1, out_cb, 1);           // DST[1] -> out[0,1]
  pack_tile<true>(2, out_cb, 2);           // DST[2] -> out[1,0]
  pack_tile<true>(3, out_cb, 3);           // DST[3] -> out[1,1]

  tile_regs_release();

  cb_push_back(out_cb, 4);
  cb_pop_front(c_cb, 4);
  cb_pop_front(b_cb, 2);
  cb_pop_front(a_cb, 2);
}
```

The Python expression `relu(a @ b + c)` compiles to a single sync region. The `+` is eliminated by pre-loading `C` into DST via `copy_tile`, then `matmul_block` accumulates `A * B` on top (`DST += A*B`). Relu operates in-place on the 4 DST registers before packing. No intermediate circular buffers or explicit add instructions are generated.

## Generating This Documentation

```bash
TTLANG_VERBOSE_PASSES=1 python test/python/test_matmul_acc.py 2>&1 > /tmp/pipeline.log
```

See `CLAUDE.md` "Workflow 1: Trace Issue Through Pass Pipeline" for details on extracting per-pass IR snapshots.
