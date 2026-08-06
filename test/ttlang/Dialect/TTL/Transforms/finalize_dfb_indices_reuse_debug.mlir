// Tests default-mode debug reporting when three logical DFBs use two physical
// indices.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' -debug-only=ttl-finalize-dfb-indices 2>&1 | FileCheck %s

// CHECK: Total DFB count: 2
// CHECK-NEXT: DFB assignment: logical DFB 0 -> physical index 0 (bounded)
// CHECK-NEXT: DFB assignment: logical DFB 1 -> physical index 1 (bounded)
// CHECK-NEXT: DFB assignment: logical DFB 2 -> physical index 0 (bounded)

module {
  func.func @producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %acknowledgment = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %first_block = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %first : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %acknowledgment_block = ttl.cb_wait %acknowledgment
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %acknowledgment : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second_block = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    return
  }

  func.func @consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %acknowledgment = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %first_block = ttl.cb_wait %first
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %first : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %acknowledgment_block = ttl.cb_reserve %acknowledgment
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %acknowledgment : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second_block = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    return
  }
}
