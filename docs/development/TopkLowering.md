# TopK Lowering

`ttl.topk` lowers to the single-core local bitonic sequence used by the metal
TopK kernels. The packed-key representation is a property of the dataflow
buffers that carry it. It is not a property of one destination-register
section.

## Representation epoch

`stable` selects fused keys. The section that first writes those keys emits
`topk_fuse`, then the local sort, then packs the keys. Later sections copy
the keys, run `topk_merge` or `topk_rebuild` with `fused` set, and pack keys
again. They do not fuse or defuse. The section that consumes the keys emits
`topk_defuse` with `num_tiles = 1`, packs plain value and index tiles, and
then transposes those tiles back to row layout. The index transpose is
followed by `topk_uint16_move_dest_tile_to_pack_half`, because a u16 transpose
leaves the datums in the low half of each 32-bit destination word.

`ttl.topk_payload` on `ttl.bind_cb` records that representation. `fused_keys`
names a buffer of `u32` tiles. `rank_stamped` names a buffer whose values
still carry rank tags. An absent attribute means plain tiles.

The sort polarity is part of the packed representation. `topk_fuse` and
`topk_stamp_local_positions` encode `largest` into the key bits, and every
later stage and `topk_defuse` must decode with the same value. `ttl.topk_order`
on a packed `ttl.bind_cb` records that polarity beside the payload, and each
stage and helper carries its own `order`. `ttl-verify-topk-epoch` requires
every `order` in one section to agree, requires a packed buffer to carry
`ttl.topk_order`, and rejects a section that reads or stores a packed buffer
of another order. The conversion to TTKernel derives `tie_order` from `order`
on a `stable_sort` stage and `largest` from `order` on the helpers; the
TTKernel verifiers reject a `tie_order` without `stable_sort` and a
`stable_sort` without `tie_order`.

Unstable mode keeps value tiles and index tiles in separate plain buffers. It
does not emit fuse or defuse.

`topk_tile_init` remains an init. `ttkernel-insert-inits` emits it from the
stage mode and does not invent helpers.

## Why this is not a dataflow-buffer transaction

`DFBValueLifetimeAnalysis` tracks when a buffer slot can be reused. Each metal
merge is its own wait, compute, and push, so tying fuse and defuse to that
transaction packs and unpacks the keys around every stage. The keys must stay
packed across those transactions. The lowering plans the epoch and places the
helpers on its edges. `ttl-verify-topk-epoch` rejects a fused or rank-stamped
stage that leaves the epoch, including a stage outside a destination-register
section and a section that both fuses and defuses.

## Algorithm

The row loop is `scf.for`. Tile loops are unrolled. Two scratch buffers
alternate: a stage reads one and packs the other, and tiles it does not update
are copied across. The initial sort transposes each input pair into
destination slots 0 and 1 (values) and 2 and 3 (indices), fuses when `stable`
is set, and runs `topk_local_sort`. `end_phase` is `log2(k) - 1`. Bitonic
`direction` starts as the opposite of `largest` and flips after a pair only
when `k` is 64. Merge `direction` follows `largest` and does not alternate.
After `log2(width)` merge/rebuild iterations the selected tiles are in the
buffer written by the initial sort.

The final extract matches the metal final kernel. Fused keys are defused into
column-layout staging buffers and then transposed into the result. Unstable
results are transposed directly from the scratch buffers. Packing back into
the buffer being read is not used.

`stable` does not select the comparator-stable network. Fused keys provide
the same tie order, and `stable_sort` stays false. `topk_canonicalize_negzero_values`
is legal only beside a comparator-stable local sort; this lowering does not
emit it.

Fused stages report a 32-bit destination requirement through
`TileExecutionInfo`. `ttl-set-compute-kernel-config` resolves it against the
kernel policy and rejects an explicit `fp32_dest_acc_en = false`; the
lowering does not rewrite that attribute. The lowering runs before
`ttl-finalize-dfb-indices`, so the scratch buffers receive logical identities
and L1 allocation entries.

A transpose result carries the input element type, which is what the
destination register holds. The transpose `output` operand names the buffer
the section packs into. In fused mode that is the `u32` key buffer, so
`transpose_wh_init` configures the packer for the keys that `topk_fuse`
produces.

## Code size

Tile loops are unrolled: every merge, rebuild, and copy-across of an
unchanged tile is its own destination-register section. The section count per
row is about `width / 2` for the initial sort plus `2 * width` per merge
iteration. At the 64-tile limit that is on the order of 800 sections, which
is a concern for kernel binary size. Hardware coverage is limited to
`width = 2`; `lower_topk.mlir` checks the merge and rebuild pairing at
`width = 8`.

## Limits

The verifier accepts `k` in {4, 8, 16, 32, 64}, a last-dimension `dim`,
`sorted = true`, and a row width that is a power of two in [2, 64] tiles.
`k` must divide the row width in elements. The merge network selects whole
tiles, so `k` below 32 runs the 32-wide network and the sorted result tile
holds the requested `k` columns first; this matches the metal host op, which
rounds `k` up to a tile before launching the kernel. Each result must be stored once
into a reserved dataflow buffer of the result shape. A multiply of a result
reads a compiler buffer filled by that store. The sequence is emitted at the
first result store, so a reserve created after the operation still dominates
the packs.

The index operand is the identity index tensor published by data movement.
The compiler does not generate that reader. Rank-stamped lowering,
`topk_stamp_tile_rank_range`, `k` above 64, multi-core merge, and
`sorted = false` are not implemented. The epoch verifier still checks
rank-stamped edges so a later lowering can use them.
