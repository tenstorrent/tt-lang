// Verifies page sizing for all six BFP formats with LLK-supported dimensions.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-validate-cb-budget{l1-budget-override=3968})'

// The six pages total exactly 3968 bytes: two 1088-byte BFP8 pages, two
// 576-byte BFP4 pages, and two 320-byte BFP2 pages.
func.func @all_bfp_formats_at_budget() {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bfp_f8>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bfp_bf8>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bfp_f4>, 1>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bfp_bf4>, 1>
  %cb4 = ttl.bind_cb {cb_index = 4, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bfp_f2>, 1>
  %cb5 = ttl.bind_cb {cb_index = 5, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bfp_bf2>, 1>
  func.return
}
