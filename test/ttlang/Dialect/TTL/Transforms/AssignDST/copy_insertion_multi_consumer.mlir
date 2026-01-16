// Summary: Test ttl.copy_dst insertion for multi-consumer values with unary consumers.
// RUN: ttlang-opt %s --ttl-assign-dst -debug-only=ttl-assign-dst --split-input-file 2>&1 | FileCheck %s --check-prefix=DEBUG
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8}),canonicalize)' --split-input-file | FileCheck %s --check-prefix=IR
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8 separate-output-region=1}),canonicalize)' --split-input-file | FileCheck %s --check-prefix=SEPARATE

#map = affine_map<(d0, d1) -> (d0, d1)>

// Test: Multi-consumer value with two unary consumers.
// Input pattern:
//   %0 = tile_mul(%in0, %in1)
//   %1 = tile_abs(%0)      // Unary consumer #1
//   %2 = tile_exp(%0)      // Unary consumer #2
//   yield %1, %2
//
// Expected: copy_dst inserted for first consumer (abs), last consumer (exp) uses original.
// This prevents abs from clobbering %0 before exp can use it.

// Verify Phase 1 debug output shows copy insertion
// DEBUG: === Phase 1: Copy Insertion ===
// DEBUG: Phase 1: Inserted copy_dst for consumer 0

// Verify max DST usage (2 inputs expire, then mul+exp merged + copy+abs merged = 2 registers)
// DEBUG: === Final DST Assignment ===
// DEBUG: Max DST usage: 2 / 8 registers

// Verify final IR has copy_dst
// IR-LABEL: func.func @multi_consumer_two_unary
// IR: ttl.compute
// IR: ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[B:.*]]: !ttcore.tile<32x32, f32>,
// IR-DAG: %[[MUL:.*]] = ttl.tile_mul %{{.*}}, %{{.*}}
// copy_dst should be inserted for first unary consumer (abs)
// IR: %[[COPY:.*]] = ttl.copy_dst %[[MUL]]
// IR: %[[ABS:.*]] = ttl.tile_abs %[[COPY]]
// Last consumer (exp) uses original
// IR: %[[EXP:.*]] = ttl.tile_exp %[[MUL]]
// SEPARATE: ttl.tile_exp {{.*}} {dst_idx = 0 : i32}
// IR: ttl.yield %[[ABS]], %[[EXP]]

func.func @multi_consumer_two_unary(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                    %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %init0 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 17, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init0_cb = ttl.attach_cb %init0, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1_cb = ttl.attach_cb %init1, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result:2 = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                         tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init0_cb, %init1_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                  tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out0_tile: !ttcore.tile<32x32, f32>,
       %out1_tile: !ttcore.tile<32x32, f32>):
    %mul = ttl.tile_mul %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %abs = ttl.tile_abs %mul : !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %mul : !ttcore.tile<32x32, f32>
    ttl.yield %abs, %exp : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>
  } -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)

  func.return %result#0, %result#1 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Test: Multi-consumer value with only binary consumers - NO copy should be inserted.
// Binary ops don't modify their inputs, so no protection needed.
// DEBUG: Max DST usage: 3 / 8 registers

// CHECK-LABEL: func.func @multi_consumer_all_binary
// CHECK: ttl.compute
// CHECK: ^bb0
// CHECK-DAG: %[[MUL:.*]] = ttl.tile_mul %{{.*}}, %{{.*}}
// No copy_dst should be inserted - all consumers are binary
// CHECK-NOT: ttl.copy_dst
// CHECK: %[[ADD1:.*]] = ttl.tile_add %[[MUL]], %{{.*}}
// SEPARATE: ttl.tile_add {{.*}} {dst_idx = 2 : i32}
// CHECK: %[[SUB:.*]] = ttl.tile_sub %[[MUL]], %{{.*}}
// SEPARATE: ttl.tile_sub {{.*}} {dst_idx = 3 : i32}
// CHECK: ttl.yield

func.func @multi_consumer_all_binary(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                     %b: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                     %c: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %init0 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb4 = ttl.bind_cb {cb_index = 17, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init0_cb = ttl.attach_cb %init0, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1_cb = ttl.attach_cb %init1, %cb4 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result:2 = ttl.compute
      ins(%a_cb, %b_cb, %c_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                tensor<2x2x!ttcore.tile<32x32, f32>>,
                                tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init0_cb, %init1_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                  tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %c_tile: !ttcore.tile<32x32, f32>,
       %out0_tile: !ttcore.tile<32x32, f32>,
       %out1_tile: !ttcore.tile<32x32, f32>):
    %mul = ttl.tile_mul %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    // Both consumers are binary - no copy needed
    %add = ttl.tile_add %mul, %c_tile : !ttcore.tile<32x32, f32>
    %sub = ttl.tile_sub %mul, %c_tile : !ttcore.tile<32x32, f32>
    ttl.yield %add, %sub : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>
  } -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)

  func.return %result#0, %result#1 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
}
