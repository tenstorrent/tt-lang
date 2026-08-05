// Tests minimum physical-index count when first-fit exceeds the L1 budget.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s

// The DFBs conflict in A-B-C-D order but are declared A,D,B,C, so first-fit
// uses three indices although two suffice. Each physical DFB occupies 491520
// bytes: three exceed the 1466368-byte fallback budget, while two fit.
// Compilation therefore requires exact search to replace the valid first-fit
// assignment with one that uses the minimum physical-index count.

// CHECK: module attributes {ttl.dfb_allocations = [{{.*}}dfb_index = 0 : i32{{.*}}, {{.*}}dfb_index = 1 : i32{{.*}}]}
// CHECK-LABEL: func.func @l1_requires_minimum_assignment
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 1 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 2 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 3 : index}

module {
  func.func @l1_requires_minimum_assignment()
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
