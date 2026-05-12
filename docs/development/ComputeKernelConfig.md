# Compute Kernel Configuration

`TTLSetComputeKernelConfig`
([source](https://github.com/tenstorrent/tt-lang/blob/main/lib/Dialect/TTL/Transforms/TTLSetComputeKernelConfig.cpp))
sets three per-kernel attributes on each `ttl.compute` function:

- `ttl.fp32_dest_acc_en` — DST holds fp32 (8 tiles) instead of bf16 (16 tiles).
- `ttl.dst_full_sync_en` — DST uses full synchronization mode.
- `ttl.unpack_to_dest_fp32` — selects `UnpackToDestMode::UnpackToDestFp32` on
  CB index 0 so f32 unpacks land in DST with full precision.

`_compile_ttnn_kernel` in `python/ttl/ttl_api.py` reads each attribute under
its prefixed name and forwards the value to `ComputeConfigDescriptor`. The
attribute names must agree between the C++ constants
(`include/ttlang/Dialect/TTL/IR/TTL.h`) and the Python lookups; a mismatch
on either side silently disables the configuration.

## Data Movement Strategies

A tile op's inputs reach the math engine via one of two strategies:

- SFPU strategy. `copy_tile` loads the input CB into DST; the SFPU then
  operates in DST in place. Used by `tile_typecast`, `tile_<unary>`, and
  SFPU binary ops.
- FPU strategy. The unpacker loads the input CB into SRCA/SRCB; the FPU
  reads from SRCA/B and writes the result to DST. Used by `tile_reduce`,
  `tile_matmul_block`, and `tile_add_fpu`/`tile_sub_fpu`/`tile_mul_fpu`.

The strategy is a structural property of the IR:

| Classification                                | Strategy |
|-----------------------------------------------|----------|
| `TTLDSTInputsTrait` and not `isCBInputOp(op)` | SFPU     |
| `TTLCBInputTileOpTrait`                     | FPU      |

`isCBInputOp(op)` (in `TTLOpsUtils.h`) returns true for ops carrying
`TTLCBInputTileOpTrait`. Polymorphic `ttl.tile_add`/`tile_sub`/`tile_mul`
(with `TTLPolymorphicBinaryTileOpTrait`) are lowered by `ttl-lower-binary-tiles`
to either `ttl.tile_*_fpu` (CB inputs) or `ttl.tile_*_sfpu` (DST inputs)
before `TTLSetComputeKernelConfig` and `TTLAssignDST` run; see
[FPU binary lowering](#fpu-binary-lowering) below.

### Dtype-Dependent Strategies

A few tt-metal LLK ops switch between strategies at runtime based on input
dtype, because SRCB is 19 bits wide and cannot hold 32-bit formats. The TTL
trait classifies the op statically; the pass does not infer
`unpack_to_dest_fp32` for these ops, so their f32 inputs lose precision
during unpack (truncated to bf16) but the FPU/SRCA strategy remains
available.

| Op               | bf16 / fp16 strategy | fp32 strategy           | Inferred here? |
|------------------|----------------------|-------------------------|----------------|
| `tile_bcast`     | FPU (SRCA/SRCB)      | SFPU (unpack-to-dest)   | no             |
| `tile_typecast`  | SFPU                 | SFPU                    | yes            |

Inferring f32 precision for `tile_bcast` requires per-CB unpack mode (see
[Current Limitation](#current-limitation)) because the affected kernels
typically mix `tile_bcast` with FPU consumers of CB 0.

## Option Semantics

### `fp32_dest_acc_en`

Kernel-global. Enabled when any tile op in the kernel reads or produces an
f32 tile: any f32 block argument; an f32 reduce qualifying under
`reduceFullFp32`; a `tile_matmul_block` under `matmulFullFp32` (suppressed
when a bf16 `tile_bcast` is also present, per a documented llk constraint).

### `dst_full_sync_en`

Kernel-global. Enabled only when the `dstFullSyncEn` pass option is true.

### `unpack_to_dest_fp32`

Kernel-global attribute with per-CB effect. The pass enables it when an
SFPU-strategy op in the compute body has an operand that is

1. an input block argument of the enclosing `ttl.compute`, and
2. of f32 tile element type.

Both conditions are necessary. Condition (1) excludes intermediate values
produced by upstream ops in the body: those operands live in DST, are not
unpacked from a CB, and so their dtype is irrelevant to the unpack mode.
Condition (2) is the precision requirement: only f32 inputs need the
non-default unpack mode.

`_set_unpack_to_dest_fp32` in `ttl_api.py` writes
`UnpackToDestMode::UnpackToDestFp32` to CB index 0 and
`UnpackToDestMode::Default` to the remaining CBs.

`UnpackToDestFp32` is incompatible with the SRCA/SRCB unpack stream (see
tt-metal `base_types.hpp`). Setting it on a CB read by an FPU-strategy op
zeros the value the FPU reads. The pass must therefore consult the
consumer's strategy, not just the dtype of the block argument. This
mirrors tt-metal's `preserve_fp32_precision` policy in
`ttnn/operations/eltwise/unary_ng/unary_ng.cpp` and
`ttnn/operations/copy/typecast/typecast.cpp`, which sets the flag only for
SFPU-strategy ops.

## FPU binary lowering

Polymorphic `ttl.tile_add`, `ttl.tile_sub`, and `ttl.tile_mul` carry
`TTLPolymorphicBinaryTileOpTrait` (and not `TTLDSTInputsTrait`) until
`TTLLowerBinaryTiles` runs. That pass replaces each op with `ttl.tile_*_fpu`
(`TTLCBInputTileOpTrait`) when `isFpuBinaryEligible` holds and
`enable-fpu-binary-ops` is true, otherwise with `ttl.tile_*_sfpu`
(`TTLDSTInputsTrait`). The pass is scheduled in `createTTLToTTKernelPipeline`
between `convert-ttl-to-compute` and the first consumer (`TTLSetComputeKernelConfig`,
then `TTLAssignDST`, then `ConvertTTLTileOpsToTTKernel`).

The pass walks each `ttl.compute` body and applies the shared predicate
`isFpuBinaryEligible` (in `TTLOpsUtils.h`). The predicate accepts
`tile_add`/`sub`/`mul` whose two operands are input block arguments of the
enclosing `ttl.compute` with matching indexing maps. The
`enable-fpu-binary-ops` option gates FPU selection: when false, eligible ops
still lower to `ttl.tile_*_sfpu` (they are never left polymorphic).

Consumers use `isCBInputOp` (trait-only) and `isFpuBinaryTileOp` where needed.
This keeps the SFPU/FPU decision a single structural property of the IR.

## Current Limitation

`unpack_to_dest_fp32` is a kernel-global boolean with a hard-coded target
of CB index 0. The current encoding is correct only when the kernel has
exactly one f32 SFPU input, on CB 0, and never mixes strategies. It does
not express:

- mixed SFPU and FPU consumers of different CBs in the same kernel,
- an SFPU-strategy f32 input on a non-zero CB index.

A per-CB `UnpackToDestMode` attribute (e.g. `DenseI32ArrayAttr`) forwarded
directly to `ComputeConfigDescriptor.unpack_to_dest_mode` covers the
general case.
