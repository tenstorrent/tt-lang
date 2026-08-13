// Tests that Blackhole accepts physical DFB index 32.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s

// CHECK: ttl.dfb_allocations = [
// CHECK-COUNT-33: dfb_index = {{[0-9]+}} : i32
// CHECK-NOT: dfb_index
// CHECK-LABEL: func.func @blackhole_accepts_33_indices
// CHECK-SAME: ttl.base_cta_index = 33 : i32
// CHECK: ttl.bind_cb{cb_index = 32, block_count = 1}

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @blackhole_accepts_33_indices()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.base_cta_index = 33 : i32, ttl.crta_indices = []} {
    %dfb0 = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb1 = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb2 = ttl.bind_cb {cb_index = 2, block_count = 1} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb3 = ttl.bind_cb {cb_index = 3, block_count = 1} {dfb_id = 3 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb4 = ttl.bind_cb {cb_index = 4, block_count = 1} {dfb_id = 4 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb5 = ttl.bind_cb {cb_index = 5, block_count = 1} {dfb_id = 5 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb6 = ttl.bind_cb {cb_index = 6, block_count = 1} {dfb_id = 6 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb7 = ttl.bind_cb {cb_index = 7, block_count = 1} {dfb_id = 7 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb8 = ttl.bind_cb {cb_index = 8, block_count = 1} {dfb_id = 8 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb9 = ttl.bind_cb {cb_index = 9, block_count = 1} {dfb_id = 9 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb10 = ttl.bind_cb {cb_index = 10, block_count = 1} {dfb_id = 10 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb11 = ttl.bind_cb {cb_index = 11, block_count = 1} {dfb_id = 11 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb12 = ttl.bind_cb {cb_index = 12, block_count = 1} {dfb_id = 12 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb13 = ttl.bind_cb {cb_index = 13, block_count = 1} {dfb_id = 13 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb14 = ttl.bind_cb {cb_index = 14, block_count = 1} {dfb_id = 14 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb15 = ttl.bind_cb {cb_index = 15, block_count = 1} {dfb_id = 15 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb16 = ttl.bind_cb {cb_index = 16, block_count = 1} {dfb_id = 16 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb17 = ttl.bind_cb {cb_index = 17, block_count = 1} {dfb_id = 17 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb18 = ttl.bind_cb {cb_index = 18, block_count = 1} {dfb_id = 18 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb19 = ttl.bind_cb {cb_index = 19, block_count = 1} {dfb_id = 19 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb20 = ttl.bind_cb {cb_index = 20, block_count = 1} {dfb_id = 20 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb21 = ttl.bind_cb {cb_index = 21, block_count = 1} {dfb_id = 21 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb22 = ttl.bind_cb {cb_index = 22, block_count = 1} {dfb_id = 22 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb23 = ttl.bind_cb {cb_index = 23, block_count = 1} {dfb_id = 23 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb24 = ttl.bind_cb {cb_index = 24, block_count = 1} {dfb_id = 24 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb25 = ttl.bind_cb {cb_index = 25, block_count = 1} {dfb_id = 25 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb26 = ttl.bind_cb {cb_index = 26, block_count = 1} {dfb_id = 26 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb27 = ttl.bind_cb {cb_index = 27, block_count = 1} {dfb_id = 27 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb28 = ttl.bind_cb {cb_index = 28, block_count = 1} {dfb_id = 28 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb29 = ttl.bind_cb {cb_index = 29, block_count = 1} {dfb_id = 29 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb30 = ttl.bind_cb {cb_index = 30, block_count = 1} {dfb_id = 30 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb31 = ttl.bind_cb {cb_index = 31, block_count = 1} {dfb_id = 31 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %dfb32 = ttl.bind_cb {cb_index = 32, block_count = 1} {dfb_id = 32 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
