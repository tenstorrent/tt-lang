// Tests logical lifecycle aggregation, safe alias preservation, and exact
// physical DFB allocation for the confirmed PR #775 review cases.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s

// Compiler-created declarations with one logical ID form one lifecycle across
// producer and consumer functions.

// CHECK-LABEL: func.func @split_compiler_producer
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 7 : index, ttl.compiler_allocated}
// CHECK-LABEL: func.func @split_compiler_consumer
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 7 : index, ttl.compiler_allocated}

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @split_compiler_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 7 : index, ttl.compiler_allocated}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_reserve %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @split_compiler_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 7 : index, ttl.compiler_allocated}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// The four data DFBs have a path interference graph. Logical-ID first-fit
// visits A, D, B, C and uses three data colors, while the exact assignment uses
// two. The greedy assignment is valid and fits the hardware limit, so the
// production allocator accepts its six total indices. The independent oracle
// joins this graph with a 30-vertex clique to verify exact escalation from 33
// to 32 indices.

// CHECK: module attributes {ttl.dfb_allocations = [{{.*}}dfb_index = 0 : i32{{.*}}, {{.*}}dfb_index = 1 : i32{{.*}}, {{.*}}dfb_index = 2 : i32{{.*}}, {{.*}}dfb_index = 3 : i32{{.*}}, {{.*}}dfb_index = 4 : i32{{.*}}, {{.*}}dfb_index = 5 : i32{{.*}}]}
// CHECK-LABEL: func.func @path_producer
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 1 : index}
// CHECK: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 2 : index}
// CHECK: ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 3 : index}
// CHECK: ttl.bind_cb{cb_index = [[ACK_AC_INDEX:[0-9]+]], block_count = 2} {dfb_id = 4 : index}
// CHECK: ttl.bind_cb{cb_index = [[ACK_AD_INDEX:[0-9]+]], block_count = 2} {dfb_id = 5 : index}
// CHECK: ttl.bind_cb{cb_index = [[ACK_BD_INDEX:[0-9]+]], block_count = 2} {dfb_id = 6 : index}

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @path_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %first_path_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %last_path_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second_path_dfb = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %third_path_dfb = ttl.bind_cb {cb_index = 3, block_count = 2} {dfb_id = 3 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %ack_ac = ttl.bind_cb {cb_index = 4, block_count = 2} {dfb_id = 4 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, f32>, 2>
    %ack_ad = ttl.bind_cb {cb_index = 5, block_count = 2} {dfb_id = 5 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, f32>, 2>
    %ack_bd = ttl.bind_cb {cb_index = 6, block_count = 2} {dfb_id = 6 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, f32>, 2>

    %first_path_slot = ttl.cb_reserve %first_path_dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %first_path_dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second_path_slot = ttl.cb_reserve %second_path_dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %second_path_dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %ack_ac_block = ttl.cb_wait %ack_ac : <[1, 1], !ttcore.tile<1x16, f32>, 2> -> tensor<1x1x!ttcore.tile<1x16, f32>>
    ttl.cb_pop %ack_ac : <[1, 1], !ttcore.tile<1x16, f32>, 2>
    %third_path_slot = ttl.cb_reserve %third_path_dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %third_path_dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %ack_ad_block = ttl.cb_wait %ack_ad : <[1, 1], !ttcore.tile<1x16, f32>, 2> -> tensor<1x1x!ttcore.tile<1x16, f32>>
    ttl.cb_pop %ack_ad : <[1, 1], !ttcore.tile<1x16, f32>, 2>
    %ack_bd_block = ttl.cb_wait %ack_bd : <[1, 1], !ttcore.tile<1x16, f32>, 2> -> tensor<1x1x!ttcore.tile<1x16, f32>>
    ttl.cb_pop %ack_bd : <[1, 1], !ttcore.tile<1x16, f32>, 2>
    %last_path_slot = ttl.cb_reserve %last_path_dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %last_path_dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    return
  }

  func.func @path_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %first_path_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %last_path_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second_path_dfb = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %third_path_dfb = ttl.bind_cb {cb_index = 3, block_count = 2} {dfb_id = 3 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %ack_ac = ttl.bind_cb {cb_index = 4, block_count = 2} {dfb_id = 4 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, f32>, 2>
    %ack_ad = ttl.bind_cb {cb_index = 5, block_count = 2} {dfb_id = 5 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, f32>, 2>
    %ack_bd = ttl.bind_cb {cb_index = 6, block_count = 2} {dfb_id = 6 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, f32>, 2>

    %first_path_value = ttl.cb_wait %first_path_dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %first_path_dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %ack_ac_block = ttl.cb_reserve %ack_ac : <[1, 1], !ttcore.tile<1x16, f32>, 2> -> tensor<1x1x!ttcore.tile<1x16, f32>>
    ttl.cb_push %ack_ac : <[1, 1], !ttcore.tile<1x16, f32>, 2>
    %second_path_value = ttl.cb_wait %second_path_dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %second_path_dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %ack_ad_block = ttl.cb_reserve %ack_ad : <[1, 1], !ttcore.tile<1x16, f32>, 2> -> tensor<1x1x!ttcore.tile<1x16, f32>>
    ttl.cb_push %ack_ad : <[1, 1], !ttcore.tile<1x16, f32>, 2>
    %ack_bd_block = ttl.cb_reserve %ack_bd : <[1, 1], !ttcore.tile<1x16, f32>, 2> -> tensor<1x1x!ttcore.tile<1x16, f32>>
    ttl.cb_push %ack_bd : <[1, 1], !ttcore.tile<1x16, f32>, 2>
    %last_path_value = ttl.cb_wait %last_path_dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %last_path_dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %third_path_value = ttl.cb_wait %third_path_dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %third_path_dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    return
  }
}
