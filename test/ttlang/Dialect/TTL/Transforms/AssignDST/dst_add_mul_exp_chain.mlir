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
// iter_index ops provide iteration coordinates for CB indexing.
// CHECK: %[[RES:.*]] = ttl.compute
// CHECK-NEXT: ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[B:.*]]: !ttcore.tile<32x32, f32>, %[[C:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:   %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:   %[[I1:.*]] = ttl.iter_index 1 : index
// CHECK-NEXT:   %[[ADD:.*]] = ttl.tile_add %[[A]], %[[B]] into dst[%c0] {ttl.fpu_binary} : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %{{.*}}, %[[DTILE:.*]] = ttl.copy_tile %[[C]][%[[I0]], %[[I1]]] into dst[%[[C1]]]
// CHECK-NEXT:   %[[MUL:.*]] = ttl.tile_mul %[[ADD]], %[[DTILE]] into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[EXP:.*]] = ttl.tile_exp %[[MUL]] into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK-NEXT:   ttl.tile_store %[[EXP]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:   ttl.yield
// CHECK-NEXT: } -> tensor<2x2x!ttcore.tile<32x32, f32>>
// CHECK-NEXT: return %[[RES]]
// SEPARATE-LABEL: func.func @add_mul_exp_chain
// SEPARATE:      %[[ADDS:.*]] = ttl.tile_add {{.*}} into dst[%c0] {ttl.fpu_binary}
// SEPARATE:      %{{.*}}, %[[DTILES:.*]] = ttl.copy_tile {{.*}}
// SEPARATE-NEXT: %[[MULS:.*]] = ttl.tile_mul %[[ADDS]], %[[DTILES]] into dst[%c2]
// SEPARATE-NEXT: %[[EXPS:.*]] = ttl.tile_exp %[[MULS]] into dst[%c2]
// SEPARATE:      ttl.tile_store
// SEPARATE-NEXT: ttl.yield
func.func @add_mul_exp_chain(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                             %b: tensor<2x2x!ttcore.tile<32x32, f32>>,
                             %c: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %out_view = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
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
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %c0 = arith.constant 0 : index
    %sum = ttl.tile_add %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %sum, %c_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %mul into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
