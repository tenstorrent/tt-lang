// Verifies that tile dimensions rejected by tt-metal are diagnosed.
// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// Dimensions not accepted by the tt-metal Tile constructor are invalid
// ttcore tile types.
func.func @invalid_tt_metal_tile() {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      // expected-error @+2 {{expected a tt-metal tile shape, got 7x13}}
      // expected-error @below {{failed to parse TTL_CircularBuffer parameter 'elementType'}}
      : !ttl.cb<[1, 1], !ttcore.tile<7x13, bf16>, 1>
  func.return
}
