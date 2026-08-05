// Tests an inconclusive L1-triggered minimum physical-index-count search.
// RUN: ttlang-opt %s --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{exact-coloring-search-limit=1})'

// The valid three-index first-fit assignment exceeds L1, while a two-index
// assignment fits. A one-state search limit cannot establish the minimum, so
// the diagnostic must distinguish an inconclusive search from L1 exhaustion.

// expected-error @below {{'builtin.module' op deterministic first-fit uses 3 physical DFB indices; exact allocation search explored 1 states and reached the 1-state limit without proving whether the allocation fits the target L1 budget; increase `exact-coloring-search-limit`}}
module {
  func.func @l1_minimum_search_limit()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 4 : i32, ttl.crta_indices = []} {
    %path_a = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 120], !ttcore.tile<32x32, bf16>, 2>
    %path_d = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 120], !ttcore.tile<32x32, bf16>, 2>
    %path_b = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 120], !ttcore.tile<32x32, bf16>, 2>
    %path_c = ttl.bind_cb {cb_index = 3, block_count = 2} {dfb_id = 3 : index} : !ttl.cb<[1, 120], !ttcore.tile<32x32, bf16>, 2>

    %path_a_output = ttl.cb_reserve %path_a : <[1, 120], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x120x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %path_a : <[1, 120], !ttcore.tile<32x32, bf16>, 2>
    %path_b_output = ttl.cb_reserve %path_b : <[1, 120], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x120x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %path_b : <[1, 120], !ttcore.tile<32x32, bf16>, 2>
    %path_a_input = ttl.cb_wait %path_a : <[1, 120], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x120x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %path_a : <[1, 120], !ttcore.tile<32x32, bf16>, 2>
    %path_c_output = ttl.cb_reserve %path_c : <[1, 120], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x120x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %path_c : <[1, 120], !ttcore.tile<32x32, bf16>, 2>
    %path_b_input = ttl.cb_wait %path_b : <[1, 120], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x120x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %path_b : <[1, 120], !ttcore.tile<32x32, bf16>, 2>
    %path_d_output = ttl.cb_reserve %path_d : <[1, 120], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x120x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %path_d : <[1, 120], !ttcore.tile<32x32, bf16>, 2>
    %path_c_input = ttl.cb_wait %path_c : <[1, 120], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x120x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %path_c : <[1, 120], !ttcore.tile<32x32, bf16>, 2>
    %path_d_input = ttl.cb_wait %path_d : <[1, 120], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x120x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %path_d : <[1, 120], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
