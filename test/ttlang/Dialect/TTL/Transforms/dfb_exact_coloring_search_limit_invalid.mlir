// Tests the distinct diagnostic for an inconclusive bounded exact search.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{exact-coloring-search-limit=1})' --verify-diagnostics

// Thirty persistent DFBs all conflict pairwise and therefore require distinct
// indices. The remaining four DFBs conflict in A-B-C-D order but are processed
// as A,D,B,C, making first-fit use three indices when two suffice. The combined
// first-fit result uses 33 indices, but an exact assignment can use 32. A
// one-state limit must report an inconclusive search, not a capacity failure.

// expected-error @below {{'builtin.module' op deterministic first-fit uses 33 physical DFB indices; exact allocation search explored 1 states and reached the 1-state limit without proving whether the allocation fits the 32-index hardware limit; increase `exact-coloring-search-limit`}}
module {
  func.func @bounded_exact_search()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 34 : i32, ttl.crta_indices = []} {
    %persistent0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent1 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent2 = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent3 = ttl.bind_cb {cb_index = 3, block_count = 2} {dfb_id = 3 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent4 = ttl.bind_cb {cb_index = 4, block_count = 2} {dfb_id = 4 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent5 = ttl.bind_cb {cb_index = 5, block_count = 2} {dfb_id = 5 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent6 = ttl.bind_cb {cb_index = 6, block_count = 2} {dfb_id = 6 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent7 = ttl.bind_cb {cb_index = 7, block_count = 2} {dfb_id = 7 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent8 = ttl.bind_cb {cb_index = 8, block_count = 2} {dfb_id = 8 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent9 = ttl.bind_cb {cb_index = 9, block_count = 2} {dfb_id = 9 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent10 = ttl.bind_cb {cb_index = 10, block_count = 2} {dfb_id = 10 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent11 = ttl.bind_cb {cb_index = 11, block_count = 2} {dfb_id = 11 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent12 = ttl.bind_cb {cb_index = 12, block_count = 2} {dfb_id = 12 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent13 = ttl.bind_cb {cb_index = 13, block_count = 2} {dfb_id = 13 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent14 = ttl.bind_cb {cb_index = 14, block_count = 2} {dfb_id = 14 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent15 = ttl.bind_cb {cb_index = 15, block_count = 2} {dfb_id = 15 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent16 = ttl.bind_cb {cb_index = 16, block_count = 2} {dfb_id = 16 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent17 = ttl.bind_cb {cb_index = 17, block_count = 2} {dfb_id = 17 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent18 = ttl.bind_cb {cb_index = 18, block_count = 2} {dfb_id = 18 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent19 = ttl.bind_cb {cb_index = 19, block_count = 2} {dfb_id = 19 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent20 = ttl.bind_cb {cb_index = 20, block_count = 2} {dfb_id = 20 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent21 = ttl.bind_cb {cb_index = 21, block_count = 2} {dfb_id = 21 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent22 = ttl.bind_cb {cb_index = 22, block_count = 2} {dfb_id = 22 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent23 = ttl.bind_cb {cb_index = 23, block_count = 2} {dfb_id = 23 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent24 = ttl.bind_cb {cb_index = 24, block_count = 2} {dfb_id = 24 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent25 = ttl.bind_cb {cb_index = 25, block_count = 2} {dfb_id = 25 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent26 = ttl.bind_cb {cb_index = 26, block_count = 2} {dfb_id = 26 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent27 = ttl.bind_cb {cb_index = 27, block_count = 2} {dfb_id = 27 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent28 = ttl.bind_cb {cb_index = 28, block_count = 2} {dfb_id = 28 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent29 = ttl.bind_cb {cb_index = 29, block_count = 2} {dfb_id = 29 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %path_a = ttl.bind_cb {cb_index = 30, block_count = 2} {dfb_id = 30 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %path_d = ttl.bind_cb {cb_index = 31, block_count = 2} {dfb_id = 31 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %path_b = ttl.bind_cb {cb_index = 32, block_count = 2} {dfb_id = 32 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %path_c = ttl.bind_cb {cb_index = 33, block_count = 2} {dfb_id = 33 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

    %reserved_a = ttl.cb_reserve %path_a : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %path_a : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reserved_b = ttl.cb_reserve %path_b : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %path_b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %waited_a = ttl.cb_wait %path_a : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %path_a : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reserved_c = ttl.cb_reserve %path_c : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %path_c : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %waited_b = ttl.cb_wait %path_b : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %path_b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reserved_d = ttl.cb_reserve %path_d : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %path_d : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %waited_c = ttl.cb_wait %path_c : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %path_c : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %waited_d = ttl.cb_wait %path_d : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %path_d : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
