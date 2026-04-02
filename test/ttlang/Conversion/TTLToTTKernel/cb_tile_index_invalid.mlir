// RUN: ttlang-opt --convert-ttl-to-ttkernel --verify-diagnostics --split-input-file %s

// Tests for tile_store index validation in convert-ttl-to-ttkernel.
// tile_store requires pre-materialized multi-dimensional indices from
// ttl-lower-to-loops. These tests verify error handling when indices
// are missing.

// -----

// tile_store with empty indices outside compute body: the AffineLinearizeIndexOp
// folds to constant 0, producing a valid pack_tile at index 0.
module {
  func.func @tile_store_outside_compute(
      %tile: !ttcore.tile<32x32, bf16>) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %fake_cb = builtin.unrealized_conversion_cast to !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
    %view = builtin.unrealized_conversion_cast %fake_cb : !ttkernel.cb<4, !ttcore.tile<32x32, bf16>> to tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.tile_store %tile, %view[] : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    func.return
  }
}
