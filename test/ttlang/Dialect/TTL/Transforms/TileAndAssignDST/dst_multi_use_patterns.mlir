// Summary: Tests for various multi-use patterns including diamond dependencies
// and fan-out scenarios to ensure the DST allocator correctly handles values
// used by multiple operations without clobbering live registers.

// RUN: ttlang-opt %s --ttl-tile-and-assign-dst --canonicalize --split-input-file | FileCheck %s

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

// 1. Copy A and B
// CHECK-NEXT: %[[TOKA:.*]], %[[TA:.*]] = ttl.copy_tile %[[A]], %[[C0:.*]], %[[C0]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK-NEXT: %[[TOKB:.*]], %[[TB:.*]] = ttl.copy_tile %[[B]], %[[C0]], %[[C1:.*]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>

// 2. Compute SUM (result in 0). Regs for A (0) and B (1) can be freed after this (if not used elsewhere).
// CHECK-NEXT: %[[SUM:.*]] = ttl.tile_add %[[TA]], %[[TB]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>

// 3. Copy C into reg 1 which is now available
// CHECK-NEXT: %[[TOKC:.*]], %[[TC:.*]] = ttl.copy_tile %[[C]], %[[C0]], %[[C1]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>

// 4. Compute DIFF (1) = SUM (0) - C (1). SUM must NOT be clobbered here (i.e. not in-place on SUM if SUM is needed later). C can be clobbered.
// CHECK-NEXT: %[[DIFF:.*]] = ttl.tile_sub %[[SUM]], %[[TC]] {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>

// 5. Copy D into reg 2 which is available
// CHECK-NEXT: %[[TOKD:.*]], %[[TD:.*]] = ttl.copy_tile %[[D]], %[[C0]], %[[C2:.*]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>

// 6. Compute PROD (0) = SUM (0) * D (2) . SUM is now last used so PROD can clobber its register.
// CHECK-NEXT: %[[PROD:.*]] = ttl.tile_mul %[[SUM]], %[[TD]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>

// 7. Compute COMBO (0) = DIFF (1) + PROD (0).
// CHECK-NEXT: %[[COMBO:.*]] = ttl.tile_add %[[DIFF]], %[[PROD]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>

// CHECK-NEXT: ttl.yield %[[COMBO]] : !ttcore.tile<32x32, f32>

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
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: ttl.compute
// CHECK-NEXT: ^bb0(%[[ARG0:.*]]: !ttcore.tile<32x32, f32>, %[[ARG1:.*]]: !ttcore.tile<32x32, f32>, %[[ARG2:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:   %[[COPY0TOK:.*]], %[[COPY0:.*]] = ttl.copy_tile %[[ARG0]], %[[C0]], %[[C0]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[COPY1TOK:.*]], %[[COPY1:.*]] = ttl.copy_tile %[[ARG1]], %[[C0]], %[[C1]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[INTERMEDIATE:.*]] = ttl.tile_add %[[COPY0]], %[[COPY1]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[COPY2TOK:.*]], %[[COPY2:.*]] = ttl.copy_tile %[[ARG2]], %[[C0]], %[[C1]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[USE1:.*]] = ttl.tile_mul %[[INTERMEDIATE]], %[[COPY2]] {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[USE2:.*]] = ttl.tile_exp %[[INTERMEDIATE]] {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[USE3:.*]] = ttl.tile_add %[[INTERMEDIATE]], %[[USE1]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[FINAL:.*]] = ttl.tile_add %[[USE3]], %[[USE2]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   ttl.yield %[[FINAL]]

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
