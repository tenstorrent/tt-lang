// Verifies that allocation with user reuse disabled leaves provisional DFB
// indices and kernel configuration unchanged and creates no runtime metadata.
// RUN: ttlang-opt %s --verify-diagnostics --mlir-print-ir-after=ttl-finalize-dfb-indices --mlir-print-ir-after-failure -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})' 2>&1 | FileCheck %s --implicit-check-not=ttl.dfb_allocations

// expected-error @below {{need 33 unspilled DFB indices but hardware supports at most 32 (1 compiler-allocated after proven reuse)}}
module {
  func.func @all_user_indices()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %user0 = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user1 = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user2 = ttl.bind_cb {cb_index = 2, block_count = 1} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user3 = ttl.bind_cb {cb_index = 3, block_count = 1} {dfb_id = 3 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user4 = ttl.bind_cb {cb_index = 4, block_count = 1} {dfb_id = 4 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user5 = ttl.bind_cb {cb_index = 5, block_count = 1} {dfb_id = 5 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user6 = ttl.bind_cb {cb_index = 6, block_count = 1} {dfb_id = 6 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user7 = ttl.bind_cb {cb_index = 7, block_count = 1} {dfb_id = 7 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user8 = ttl.bind_cb {cb_index = 8, block_count = 1} {dfb_id = 8 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user9 = ttl.bind_cb {cb_index = 9, block_count = 1} {dfb_id = 9 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user10 = ttl.bind_cb {cb_index = 10, block_count = 1} {dfb_id = 10 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user11 = ttl.bind_cb {cb_index = 11, block_count = 1} {dfb_id = 11 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user12 = ttl.bind_cb {cb_index = 12, block_count = 1} {dfb_id = 12 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user13 = ttl.bind_cb {cb_index = 13, block_count = 1} {dfb_id = 13 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user14 = ttl.bind_cb {cb_index = 14, block_count = 1} {dfb_id = 14 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user15 = ttl.bind_cb {cb_index = 15, block_count = 1} {dfb_id = 15 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user16 = ttl.bind_cb {cb_index = 16, block_count = 1} {dfb_id = 16 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user17 = ttl.bind_cb {cb_index = 17, block_count = 1} {dfb_id = 17 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user18 = ttl.bind_cb {cb_index = 18, block_count = 1} {dfb_id = 18 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user19 = ttl.bind_cb {cb_index = 19, block_count = 1} {dfb_id = 19 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user20 = ttl.bind_cb {cb_index = 20, block_count = 1} {dfb_id = 20 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user21 = ttl.bind_cb {cb_index = 21, block_count = 1} {dfb_id = 21 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user22 = ttl.bind_cb {cb_index = 22, block_count = 1} {dfb_id = 22 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user23 = ttl.bind_cb {cb_index = 23, block_count = 1} {dfb_id = 23 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user24 = ttl.bind_cb {cb_index = 24, block_count = 1} {dfb_id = 24 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user25 = ttl.bind_cb {cb_index = 25, block_count = 1} {dfb_id = 25 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user26 = ttl.bind_cb {cb_index = 26, block_count = 1} {dfb_id = 26 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user27 = ttl.bind_cb {cb_index = 27, block_count = 1} {dfb_id = 27 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user28 = ttl.bind_cb {cb_index = 28, block_count = 1} {dfb_id = 28 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user29 = ttl.bind_cb {cb_index = 29, block_count = 1} {dfb_id = 29 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user30 = ttl.bind_cb {cb_index = 30, block_count = 1} {dfb_id = 30 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %user31 = ttl.bind_cb {cb_index = 31, block_count = 1} {dfb_id = 31 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }

  // CHECK-LABEL: func.func @compiler_dfb
  // CHECK-SAME: ttl.base_cta_index = 1 : i32
  // CHECK: ttl.bind_cb{cb_index = 0, block_count = 1} {ttl.compiler_allocated}
  func.func @compiler_dfb()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 1 : i32} {
    %compiler_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {ttl.compiler_allocated}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %reserve = ttl.cb_reserve %compiler_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %compiler_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %wait = ttl.cb_wait %compiler_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %compiler_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
