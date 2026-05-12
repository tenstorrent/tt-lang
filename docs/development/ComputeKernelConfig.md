# Compute Kernel Configuration

The `TTLSetComputeKernelConfig` pass
([source](https://github.com/tenstorrent/tt-lang/blob/main/lib/Dialect/TTL/Transforms/TTLSetComputeKernelConfig.cpp))
sets per-kernel options on a `ttl.compute` function:

- `fp32_dest_acc_en` — DST holds fp32 (8 tiles) instead of bf16/fp16 (16 tiles).
- `dst_full_sync_en` — DST uses full synchronization mode.
- `ttl.unpack_to_dest_fp32` — input CB(s) need `UnpackToDestFp32` unpack mode.

`_compile_ttnn_kernel` in `python/ttl/ttl_api.py` reads the attributes back and
forwards them to `ComputeConfigDescriptor`.

## Data Movement Strategies

A tile op's inputs reach the math engine via one of two strategies:

- SFPU strategy: `copy_tile` loads the input CB into DST; the SFPU then
  operates in DST in place. Used by `tile_typecast`, `tile_<unary>`, and SFPU
  binary ops.
- FPU strategy: the unpacker loads the input CB into SRCA/SRCB; the FPU reads
  from SRCA/B and writes the result to DST. Used by `tile_reduce`,
  `tile_matmul_block`, and ops marked `ttl.fpu_binary`.

In TTL the strategy is determined by op trait:

| Trait / attribute                                  | Strategy | Examples                                   |
|----------------------------------------------------|----------|--------------------------------------------|
| `TTLDSTInputsTrait` and not `isCBInputOp(...)`     | SFPU     | `tile_typecast`, `tile_<unary>`, SFPU bins |
| `TTLCBInputTileOpTrait` or `kFPUBinaryAttrName`    | FPU      | `tile_reduce`, `tile_matmul_block`, ...    |

`isCBInputOp(op)` returns true for both `TTLCBInputTileOpTrait` ops and ops
marked `ttl.fpu_binary` (FPU-mode binaries override the default SFPU
classification via an attribute, not a trait).

### Dtype-Dependent Strategies

A few tt-metal LLK ops switch between strategies at runtime based on the input
dtype, because the SRCB register is only 19 bits wide and cannot hold 32-bit
formats. The TTL trait classifies the op statically; the pass currently does
not infer `unpack_to_dest_fp32` for these ops, so their f32 inputs lose
precision during unpack (truncated to bf16) but the FPU/SRCA path remains
available.

| Op               | bf16 / fp16 strategy | fp32 strategy           | Inferred here? |
|------------------|----------------------|-------------------------|----------------|
| `tile_bcast`     | FPU (SRCA/SRCB)      | SFPU (unpack-to-dest)   | no             |
| `tile_typecast`  | SFPU                 | SFPU                    | yes            |

Adding f32 precision for `tile_bcast` requires per-CB unpack mode (see
Limitation below) because the kernels in question typically mix `tile_bcast`
with FPU consumers of CB 0.

## Option Semantics

### `fp32_dest_acc_en`

Kernel-global. Enabled when any tile op in the kernel reads or produces an f32
tile: any f32 block argument; an f32 reduce qualifying under `reduceFullFp32`;
a `tile_matmul_block` under `matmulFullFp32` (suppressed when a bf16
`tile_bcast` is also present, per a documented llk constraint).

### `dst_full_sync_en`

Kernel-global. Enabled only when the `dstFullSyncEn` pass option is true.

### `unpack_to_dest_fp32`

Kernel-global attribute, per-CB effect. Enabled when the body contains an
SFPU-strategy op with an f32 tile input. The pass classifies an op as
SFPU-strategy when it has `TTLDSTInputsTrait`, is not `isCBInputOp`, and is
not `isFpuBinaryEligible`.

The third check is needed because tile add/sub/mul ops carry
`TTLDSTInputsTrait` by default but `TTLAssignDST` later marks the
FPU-eligible ones (both operands are matching input block args) with
`kFPUBinaryAttrName`. Without the predicate, the pass would over-trigger
`unpack_to_dest_fp32` for kernels whose tile add/sub/mul will resolve to FPU
binaries. `isFpuBinaryEligible` is the shared predicate in
`include/ttlang/Dialect/TTL/IR/TTLOpsUtils.h` used by both passes;
`enableFPUBinaryOps` must be threaded identically through both.

`_set_unpack_to_dest_fp32` in `ttl_api.py` configures
`UnpackToDestMode::UnpackToDestFp32` on CB index 0 and
`UnpackToDestMode::Default` on the remaining CBs.

A CB configured `UnpackToDestFp32` cannot be unpacked to SRCA/B (see
tt-metal `base_types.hpp`). Setting it for an FPU-strategy input CB silently
produces all-zero compute results. The predicate must therefore be the
consumer strategy, not the input dtype alone. This mirrors tt-metal's policy
in `ttnn/operations/eltwise/unary_ng/unary_ng.cpp` and
`ttnn/operations/copy/typecast/typecast.cpp`, where `preserve_fp32_precision`
is set only by SFPU-strategy ops.

#### Pass-Order Caveat

`kFPUBinaryAttrName` is written by `TTLAssignDST`, which runs after this
pass. To classify tile add/sub/mul correctly, this pass re-evaluates the
same predicate (`isFpuBinaryEligible`) that `TTLAssignDST` uses. The shared
predicate lives in `TTLOpsUtils.h` and both passes consume the same
`enableFPUBinaryOps` value (threaded through the pipeline). Moving FPU
binary marking into a separate earlier pass would remove the duplication
and is a candidate refactor.

## Current Limitation

The attribute and the Python helper are kernel-global with a hard-coded target
of CB index 0. This is correct when the kernel has exactly one f32 SFPU input
on CB 0 and never mixes strategies. It does not express:

- mixed SFPU and FPU consumers of different CBs in the same kernel,
- an SFPU-strategy f32 input on a non-zero CB index.

A per-CB `UnpackToDestMode` attribute (e.g. `DenseI32ArrayAttr`) forwarded
directly to `ComputeConfigDescriptor.unpack_to_dest_mode` covers the general
case.
