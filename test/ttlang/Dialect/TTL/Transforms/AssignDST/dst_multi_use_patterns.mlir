// Summary: Tests for various multi-use patterns including diamond dependencies
// and fan-out scenarios to ensure the DST allocator correctly handles values
// used by multiple operations without clobbering live registers.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8}),canonicalize)' --split-input-file | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8 separate-output-region=1}),canonicalize)' --split-input-file | FileCheck %s --check-prefix=SEPARATE

// Verify no placeholder copies remain in final IR
// CHECK-NOT: placeholder

// Test: Diamond dependency pattern with intermediate result reuse.
// Purpose: Verify that a value used by multiple tile ops is copied once and
// remains valid across both uses (no clobbering of live DST registers).
// Pattern:
//   sum = add(a, b)
//   diff = sub(sum, c)
//   prod = mul(sum, d)
//   combo = add(diff, prod)
//
// 'sum' is used by both 'sub' and 'mul'. It must stay live in a register
// until both are done.

// CHECK-LABEL: func.func @diamond_intermediate_reuse
// CHECK: %[[RES:.*]] = ttl.compute
// CHECK: ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[B:.*]]: !ttcore.tile<32x32, f32>, %[[C:.*]]: !ttcore.tile<32x32, f32>, %[[D:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// Copies at first use: A and B before add, C before sub, D before mul
// CHECK:      %[[TOKA:.*]], %[[TA:.*]] = ttl.copy_tile %[[A]]
// CHECK:      %[[TOKB:.*]], %[[TB:.*]] = ttl.copy_tile %[[B]]
// CHECK:      %[[SUM:.*]] = ttl.tile_add %[[TA]], %[[TB]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK:      %{{.*}}, %[[TC:.*]] = ttl.copy_tile %[[C]]
// CHECK:      %[[DIFF:.*]] = ttl.tile_sub %[[SUM]], %[[TC]] {dst_idx = {{[0-9]+}} : i32}
// CHECK:      %{{.*}}, %[[TD:.*]] = ttl.copy_tile %[[D]]
// CHECK:      %[[PROD:.*]] = ttl.tile_mul %[[SUM]], %[[TD]] {dst_idx = 0 : i32}
// CHECK:      %[[COMBO:.*]] = ttl.tile_add %{{.*}}, %[[PROD]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// SEPARATE:   ttl.tile_add {{.*}} {dst_idx = 3 : i32}
// CHECK:      ttl.yield %[[COMBO]] : !ttcore.tile<32x32, f32>

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @diamond_intermediate_reuse(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                      %b: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                      %c: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                      %d: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb4 = ttl.bind_cb {cb_index = 4, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %d_cb = ttl.attach_cb %d, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb4 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb, %b_cb, %c_cb, %d_cb :
          tensor<2x2x!ttcore.tile<32x32, f32>>,
          tensor<2x2x!ttcore.tile<32x32, f32>>,
          tensor<2x2x!ttcore.tile<32x32, f32>>,
          tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %c_tile: !ttcore.tile<32x32, f32>,
       %d_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):

    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %diff = ttl.tile_sub %sum, %c_tile : !ttcore.tile<32x32, f32>
    %prod = ttl.tile_mul %sum, %d_tile : !ttcore.tile<32x32, f32>
    %combo = ttl.tile_add %diff, %prod : !ttcore.tile<32x32, f32>

    ttl.yield %combo : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Fan-out pattern with intermediate result consumed by multiple ops.
// Purpose: One copy per input; INTERMEDIATE stays live across mul/exp/add without a second copy.

#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @intermediate_result_fan_out
// CHECK: ttl.compute
// CHECK: ^bb0(%[[ARG0:.*]]: !ttcore.tile<32x32, f32>, %[[ARG1:.*]]: !ttcore.tile<32x32, f32>, %[[ARG2:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// Copies at first use: ARG0 and ARG1 before add, ARG2 before mul
// CHECK:      %[[COPY0TOK:.*]], %[[COPY0:.*]] = ttl.copy_tile %[[ARG0]]
// CHECK:      %[[COPY1TOK:.*]], %[[COPY1:.*]] = ttl.copy_tile %[[ARG1]]
// CHECK:      %[[INTERMEDIATE:.*]] = ttl.tile_add %[[COPY0]], %[[COPY1]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK:      ttl.tile_mul {{.*}} {dst_idx = {{[0-9]+}} : i32}
// CHECK:      ttl.tile_exp {{.*}} {dst_idx = {{[0-9]+}} : i32}
// CHECK:      ttl.tile_add {{.*}} {dst_idx = 0 : i32}
// CHECK:      %[[FINAL:.*]] = ttl.tile_add {{.*}} {dst_idx = 0 : i32}
// SEPARATE:   ttl.tile_add {{.*}} {dst_idx = 3 : i32}
// CHECK:      ttl.yield %[[FINAL]]

func.func @intermediate_result_fan_out(%i0: tensor<32x32xf32>, %i1: tensor<32x32xf32>, %i2: tensor<32x32xf32>)
    -> tensor<32x32xf32> {
  %init = tensor.empty() : tensor<32x32xf32>

  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>

  %t0 = ttl.attach_cb %i0, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t1 = ttl.attach_cb %i1, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t2 = ttl.attach_cb %i2, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t_init = ttl.attach_cb %init, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>

  %res = ttl.compute
    ins(%t0, %t1, %t2 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%t_init : tensor<32x32xf32>)
    {indexing_maps = [#map1, #map1, #map1, #map1],
     iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>,
       %arg2: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):

    // Compute an intermediate result
    %intermediate = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>

    // Fan out: intermediate result is consumed by three different ops
    // (two binary, one unary)
    %use1 = ttl.tile_mul %intermediate, %arg2 : !ttcore.tile<32x32, f32>
    %use2 = ttl.tile_exp %intermediate : !ttcore.tile<32x32, f32>
    %use3 = ttl.tile_add %intermediate, %use1 : !ttcore.tile<32x32, f32>
    %final = ttl.tile_add %use3, %use2 : !ttcore.tile<32x32, f32>

    ttl.yield %final : !ttcore.tile<32x32, f32>
  } -> tensor<32x32xf32>

  func.return %res : tensor<32x32xf32>
}
