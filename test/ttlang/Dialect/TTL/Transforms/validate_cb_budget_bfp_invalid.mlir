// Verifies BFP page sizes contribute their exact byte counts to the DFB budget.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-validate-cb-budget{l1-budget-override=3968})'

// Double-buffering the final 320-byte page raises the total to 4288 bytes.
func.func @all_bfp_formats_over_budget() {
  // expected-error @below {{total DFB allocation (4288 bytes) exceeds L1 budget (3968 bytes)}}
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bfp_f8>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bfp_bf8>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bfp_f4>, 1>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bfp_bf4>, 1>
  %cb4 = ttl.bind_cb {cb_index = 4, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bfp_f2>, 1>
  %cb5 = ttl.bind_cb {cb_index = 5, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bfp_bf2>, 2>
  func.return
}
