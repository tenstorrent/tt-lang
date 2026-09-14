// Explicit user protocols retain individual acquires; compiler intermediates still coalesce.
// RUN: ttlang-opt %s --ttl-coalesce-dfb-acquires --split-input-file | FileCheck %s --check-prefixes=COMMON,AUTO
// RUN: ttlang-opt %s --ttl-coalesce-dfb-acquires='sync-user-dfbs=false' --split-input-file | FileCheck %s --check-prefixes=COMMON,EXPLICIT

// Producer groups distinguish user declarations from compiler-created storage.
// COMMON-LABEL: func.func @producer_groups
// COMMON: %[[USER:.*]] = ttl.bind_cb
// COMMON: %[[COMPILER:.*]] = ttl.bind_cb
// AUTO: ttl.cb_reserve %[[USER]] {num_tiles = 2 : i64}
// AUTO: ttl.cb_push %[[USER]] {num_tiles = 2 : i64}
// EXPLICIT: %[[FIRST:.*]] = ttl.cb_reserve %[[USER]] :
// EXPLICIT-NEXT: %[[SECOND:.*]] = ttl.cb_reserve %[[USER]] :
// EXPLICIT-NEXT: ttl.store {{.*}}, %[[FIRST]] :
// EXPLICIT-NEXT: ttl.cb_push %[[USER]] :
// EXPLICIT-NEXT: ttl.store {{.*}}, %[[SECOND]] :
// EXPLICIT-NEXT: ttl.cb_push %[[USER]] :
// COMMON: ttl.cb_reserve %[[COMPILER]] {num_tiles = 2 : i64}
// COMMON: ttl.cb_push %[[COMPILER]] {num_tiles = 2 : i64}
// COMMON-NOT: ttl.cb_reserve
// COMMON-NOT: ttl.cb_push
// COMMON: return
func.func @producer_groups(%input_value: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %user_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %compiler_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %first_user = ttl.cb_reserve %user_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %second_user = ttl.cb_reserve %user_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %input_value, %first_user : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %user_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.store %input_value, %second_user : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %user_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %first_compiler = ttl.cb_reserve %compiler_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %second_compiler = ttl.cb_reserve %compiler_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %input_value, %first_compiler : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %compiler_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.store %input_value, %second_compiler : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %compiler_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Consumer groups apply the same ownership policy to wait/pop operations.
// COMMON-LABEL: func.func @consumer_groups
// COMMON: %[[USER:.*]] = ttl.bind_cb
// COMMON: %[[COMPILER:.*]] = ttl.bind_cb
// AUTO: ttl.cb_wait %[[USER]] {num_tiles = 2 : i64}
// AUTO: ttl.cb_pop %[[USER]] {num_tiles = 2 : i64}
// EXPLICIT: %[[FIRST:.*]] = ttl.cb_wait %[[USER]] :
// EXPLICIT-NEXT: %[[SECOND:.*]] = ttl.cb_wait %[[USER]] :
// EXPLICIT-NEXT: ttl.add %[[FIRST]], %[[SECOND]] :
// EXPLICIT-NEXT: ttl.cb_pop %[[USER]] :
// EXPLICIT-NEXT: ttl.cb_pop %[[USER]] :
// COMMON: ttl.cb_wait %[[COMPILER]] {num_tiles = 2 : i64}
// COMMON: ttl.cb_pop %[[COMPILER]] {num_tiles = 2 : i64}
// COMMON-NOT: ttl.cb_wait
// COMMON-NOT: ttl.cb_pop
// COMMON: return
func.func @consumer_groups() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %user_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %compiler_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %first_user = ttl.cb_wait %user_dfb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %second_user = ttl.cb_wait %user_dfb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %user_sum = ttl.add %first_user, %second_user : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.cb_pop %user_dfb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.cb_pop %user_dfb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  %first_compiler = ttl.cb_wait %compiler_dfb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %second_compiler = ttl.cb_wait %compiler_dfb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %compiler_sum = ttl.add %first_compiler, %second_compiler : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.cb_pop %compiler_dfb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.cb_pop %compiler_dfb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  return
}
