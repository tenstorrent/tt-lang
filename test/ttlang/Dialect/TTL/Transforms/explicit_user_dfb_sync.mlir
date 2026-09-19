// User DFB synchronization is optional; compiler-created DFBs remain automatic.
// RUN: ttlang-opt %s --ttl-insert-cb-sync --split-input-file | FileCheck %s --check-prefixes=COMMON,AUTO
// RUN: ttlang-opt %s --ttl-insert-cb-sync='sync-user-dfbs=true' --split-input-file | FileCheck %s --check-prefixes=COMMON,AUTO
// RUN: ttlang-opt %s --ttl-insert-cb-sync='sync-user-dfbs=false' --split-input-file | FileCheck %s --check-prefixes=COMMON,EXPLICIT

// Release insertion distinguishes declarations using the existing ownership marker.
// COMMON-LABEL: func.func @producer_releases
// COMMON: %[[USER:.*]] = ttl.bind_cb
// COMMON: %[[COMPILER:.*]] = ttl.bind_cb
// COMMON: ttl.cb_reserve %[[USER]]
// COMMON-NEXT: ttl.store
// AUTO-NEXT: ttl.cb_push %[[USER]]
// EXPLICIT-NOT: ttl.cb_push %[[USER]]
// COMMON: ttl.cb_reserve %[[COMPILER]]
// COMMON-NEXT: ttl.store
// COMMON-NEXT: ttl.cb_push %[[COMPILER]]
// COMMON-NEXT: return
func.func @producer_releases(%input_value: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %user_dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %compiler_dfb = ttl.bind_cb {cb_index = 1, block_count = 1} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %user_slot = ttl.cb_reserve %user_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %input_value, %user_slot : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  %compiler_slot = ttl.cb_reserve %compiler_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %input_value, %compiler_slot : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// Consumer release insertion uses the same policy, including FP32 buffers.
// COMMON-LABEL: func.func @consumer_releases
// COMMON: %[[USER:.*]] = ttl.bind_cb
// COMMON: %[[COMPILER:.*]] = ttl.bind_cb
// COMMON: ttl.cb_wait %[[USER]]
// COMMON: ttl.add
// AUTO-NEXT: ttl.cb_pop %[[USER]]
// EXPLICIT-NOT: ttl.cb_pop %[[USER]]
// COMMON: ttl.cb_wait %[[COMPILER]]
// COMMON: ttl.add
// COMMON-NEXT: ttl.cb_pop %[[COMPILER]]
// COMMON-NEXT: return
func.func @consumer_releases(%input_value: tensor<1x1x!ttcore.tile<32x32, f32>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %user_dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %compiler_dfb = ttl.bind_cb {cb_index = 1, block_count = 1} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %user_slot = ttl.cb_wait %user_dfb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %user_value = ttl.attach_cb %user_slot, %user_dfb : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %user_sum = ttl.add %user_value, %input_value : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %compiler_slot = ttl.cb_wait %compiler_dfb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %compiler_value = ttl.attach_cb %compiler_slot, %compiler_dfb : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %compiler_sum = ttl.add %compiler_value, %input_value : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  return
}
