// Verifies BFP subtiles align exponent bytes before DFB budget accounting.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-validate-cb-budget{l1-budget-override=255})'

// The target allocator rounds the 32-, 24-, and 80-byte DFBs to 64, 64, and
// 128 bytes respectively.
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
func.func @bfp_subtile_page_sizes() {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bfp_bf8>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bfp_bf4>, 1>
  // expected-error @below {{total DFB allocation (256 bytes) exceeds L1 budget (255 bytes)}}
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<2x32, bfp_bf8>, 1>
  func.return
}
}
