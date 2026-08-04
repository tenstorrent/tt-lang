// Verifies that DFB bindings reject tile dimensions outside current compute
// support and that invalid tt-metal tile dimensions are diagnosed.
// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// A tt-metal tile shape that is reserved for host loopback cannot back an LLK
// dataflow buffer.
func.func @host_loopback_tile() {
  // expected-error @below {{'ttl.bind_cb' op tile shape 8x32 is not supported by the LLK; supported shapes are 16x16, 16x32, 32x16, and 32x32}}
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>
  func.return
}

// -----

// Dimensions not accepted by the tt-metal Tile constructor are invalid
// ttcore tile types.
func.func @invalid_tt_metal_tile() {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      // expected-error @+2 {{expected a tt-metal tile shape, got 7x13}}
      // expected-error @below {{failed to parse TTL_CircularBuffer parameter 'elementType'}}
      : !ttl.cb<[1, 1], !ttcore.tile<7x13, bf16>, 1>
  func.return
}

// -----

// TT-Lang accepts BFP_Float8 compute tiles only at the default dimensions.
func.func @bfp_f8_subtile() {
  // expected-error @below {{'ttl.bind_cb' op TT-Lang supports BFP compute tiles only with 32x32 dimensions, got 16x32}}
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, bfp_f8>, 1>
  func.return
}

// -----

// TT-Lang accepts BFP_BFloat8 compute tiles only at the default dimensions.
func.func @bfp_bf8_subtile() {
  // expected-error @below {{'ttl.bind_cb' op TT-Lang supports BFP compute tiles only with 32x32 dimensions, got 16x32}}
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, bfp_bf8>, 1>
  func.return
}

// -----

// TT-Lang accepts BFP_Float4 compute tiles only at the default dimensions.
func.func @bfp_f4_subtile() {
  // expected-error @below {{'ttl.bind_cb' op TT-Lang supports BFP compute tiles only with 32x32 dimensions, got 16x32}}
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, bfp_f4>, 1>
  func.return
}

// -----

// TT-Lang accepts BFP_BFloat4 compute tiles only at the default dimensions.
func.func @bfp_bf4_subtile() {
  // expected-error @below {{'ttl.bind_cb' op TT-Lang supports BFP compute tiles only with 32x32 dimensions, got 16x32}}
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, bfp_bf4>, 1>
  func.return
}

// -----

// TT-Lang accepts BFP_Float2 compute tiles only at the default dimensions.
func.func @bfp_f2_subtile() {
  // expected-error @below {{'ttl.bind_cb' op TT-Lang supports BFP compute tiles only with 32x32 dimensions, got 16x32}}
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, bfp_f2>, 1>
  func.return
}

// -----

// TT-Lang accepts BFP_BFloat2 compute tiles only at the default dimensions.
func.func @bfp_bf2_subtile() {
  // expected-error @below {{'ttl.bind_cb' op TT-Lang supports BFP compute tiles only with 32x32 dimensions, got 16x32}}
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, bfp_bf2>, 1>
  func.return
}
