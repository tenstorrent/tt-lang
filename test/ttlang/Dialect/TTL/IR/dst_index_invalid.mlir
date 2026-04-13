// Verify that tile ops reject invalid dst_index operand types.

// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// -----

// tile_store with non-index dst_index type.
func.func @tile_store_bad_dst_type(%tile: !ttcore.tile<32x32, bf16>,
                                    %view: tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                    %idx: index,
                                    %bad: i64) {
  // expected-error @below {{operand #4 must be index, but got 'i64'}}
  "ttl.tile_store"(%tile, %view, %idx, %idx, %bad) : (!ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>, index, index, i64) -> ()
  return
}
