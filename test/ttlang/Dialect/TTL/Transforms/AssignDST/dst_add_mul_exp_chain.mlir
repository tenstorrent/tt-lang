// Summary: three-op chain (add -> mul -> exp) with DST register reuse.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8}),canonicalize,cse)' --split-input-file | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8 separate-output-region=1}),canonicalize,cse)' --split-input-file | FileCheck %s --check-prefix=SEPARATE

// Verify no placeholder copies remain in final IR
// CHECK-NOT: placeholder

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: verify FPU binary for add (both block args), then SFPU for mul/exp.
// Add reads A,B from CB (FPU). Mul reads add result from DST + C from CB (SFPU).
// Only C needs copy_tile.

// CHECK-LABEL: func.func @add_mul_exp_chain
// CHECK-SAME: (%[[AARG:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[BARG:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[CARG:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>)
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[CB0:.*]] = ttl.bind_cb
// CHECK-DAG: %[[CB1:.*]] = ttl.bind_cb
// CHECK-DAG: %[[CB2:.*]] = ttl.bind_cb
// CHECK-DAG: %[[CB3:.*]] = ttl.bind_cb
// CHECK-DAG: %[[A_CB:.*]] = ttl.attach_cb %[[AARG:.*]], %[[CB0]]
// CHECK-DAG: %[[B_CB:.*]] = ttl.attach_cb %[[BARG:.*]], %[[CB1]]
// CHECK-DAG: %[[C_CB:.*]] = ttl.attach_cb %[[CARG:.*]], %[[CB2]]
// CHECK-DAG: %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT:.*]], %[[CB3]]
// FPU binary add (no copies for A, B), then copy C for SFPU mul.
// CHECK: %[[RES:.*]] = ttl.compute
// CHECK-NEXT: ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[B:.*]]: !ttcore.tile<32x32, f32>, %[[C:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:   %[[ADD:.*]] = ttl.tile_add %[[A]], %[[B]] {dst_idx = 0 : i32, ttl.fpu_binary} : !ttcore.tile<32x32, f32>
// CHECK:        %{{.*}}, %[[DTILE:.*]] = ttl.copy_tile %[[C]], %{{.*}}, %{{.*}} {dst_idx = 1 : i32}
// CHECK-NEXT:   %[[MUL:.*]] = ttl.tile_mul %[[ADD]], %[[DTILE]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[EXP:.*]] = ttl.tile_exp %[[MUL]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   ttl.yield %[[EXP]] : !ttcore.tile<32x32, f32>
// CHECK-NEXT: } -> tensor<2x2x!ttcore.tile<32x32, f32>>
// CHECK-NEXT: return %[[RES]]
// SEPARATE-LABEL: func.func @add_mul_exp_chain
// SEPARATE:      %[[ADDS:.*]] = ttl.tile_add {{.*}} {dst_idx = 0 : i32, ttl.fpu_binary}
// SEPARATE:      %{{.*}}, %[[DTILES:.*]] = ttl.copy_tile {{.*}} {dst_idx = 1 : i32}
// SEPARATE-NEXT: %[[MULS:.*]] = ttl.tile_mul %[[ADDS]], %[[DTILES]] {dst_idx = 2 : i32}
// SEPARATE-NEXT: %[[EXPS:.*]] = ttl.tile_exp %[[MULS]] {dst_idx = 2 : i32}
// SEPARATE-NEXT: ttl.yield %[[EXPS]]
func.func @add_mul_exp_chain(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                             %b: tensor<2x2x!ttcore.tile<32x32, f32>>,
                             %c: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb, %b_cb, %c_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                tensor<2x2x!ttcore.tile<32x32, f32>>,
                                tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %c_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %sum, %c_tile : !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %mul : !ttcore.tile<32x32, f32>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
