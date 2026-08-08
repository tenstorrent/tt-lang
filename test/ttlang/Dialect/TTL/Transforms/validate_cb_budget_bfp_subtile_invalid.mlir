// Verifies BFP subtiles align exponent bytes before DFB budget accounting.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-validate-cb-budget{l1-budget-override=135})'

// The pages use 16-byte-aligned exponent sections: 32 + 24 + 80 = 136 bytes.
func.func @bfp_subtile_page_sizes() {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bfp_bf8>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bfp_bf4>, 1>
  // expected-error @below {{total circular buffer allocation (136 bytes) exceeds L1 budget (135 bytes)}}
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<2x32, bfp_bf8>, 1>
  func.return
}
