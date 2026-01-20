// Summary: Seven-operation fused chain to verify DST allocation handles long chains.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8},ttl-insert-tile-regs-sync))' | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8 separate-output-region=1},ttl-insert-tile-regs-sync))' | FileCheck %s --check-prefix=SEPARATE

// Verify no placeholder copies remain in final IR
// CHECK-NOT: placeholder

// Purpose: Regression test for DST register allocation bug where operations were
// dropped in fused chains due to register conflicts. This test verifies that all
// seven operations in the chain receive dst_idx attributes and appear in output.
// The chain: add → sub → mul → div → exp → log → relu should all be present.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL:   func.func @seven_op_chain
// CHECK:           %[[CB0:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 1}
// CHECK:           %[[CB1:.*]] = ttl.bind_cb{cb_index = 1, buffer_factor = 1}
// CHECK:           %[[CB2:.*]] = ttl.bind_cb{cb_index = 2, buffer_factor = 1}
// CHECK:           %[[RES:.*]] = ttl.compute
// CHECK:           ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[B:.*]]: !ttcore.tile<32x32, f32>, %[[O:.*]]: !ttcore.tile<32x32, f32>):
// CHECK:             ttl.tile_regs_acquire
// CHECK:             %[[LIN_IDX_0:.*]] = ttl.linearized_index #{{.*}} : index
// CHECK:             %[[DTOK0:.*]], %[[DTILE0:.*]] = ttl.copy_tile %[[A]], %[[LIN_IDX_0]], %{{.*}} : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK:             %[[LIN_IDX_1:.*]] = ttl.linearized_index #{{.*}} : index
// CHECK:             %[[DTOK1:.*]], %[[DTILE1:.*]] = ttl.copy_tile %[[B]], %[[LIN_IDX_1]], %{{.*}} : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK-NEXT:        %[[ADD:.*]] = ttl.tile_add %[[DTILE0]], %[[DTILE1]] {dst_idx = 0 : i32}
// CHECK-NEXT:        %[[SUB:.*]] = ttl.tile_sub %[[ADD]], %[[DTILE1]] {dst_idx = 0 : i32}
// CHECK-NEXT:        %[[MUL:.*]] = ttl.tile_mul %[[SUB]], %[[DTILE1]] {dst_idx = 0 : i32}
// CHECK-NEXT:        %[[EXP:.*]] = ttl.tile_exp %[[MUL]] {dst_idx = 0 : i32}
// CHECK-NEXT:        %[[LOG:.*]] = ttl.tile_log %[[EXP]] {dst_idx = 0 : i32}
// CHECK-NEXT:        %[[NEG:.*]] = ttl.tile_neg %[[LOG]] {dst_idx = 0 : i32}
// CHECK-NEXT:        %[[SQRT:.*]] = ttl.tile_sqrt %[[NEG]] {dst_idx = 0 : i32}
// SEPARATE: ttl.tile_sqrt {{.*}} {dst_idx = 2 : i32}
// CHECK-NEXT:        ttl.tile_regs_commit
// CHECK-NEXT:        ttl.tile_regs_wait
// CHECK-NEXT:        %[[VIEW:.*]] = ttl.cb_reserve %[[CB2]]
// CHECK-NEXT:        ttl.store %[[SQRT]], %[[VIEW]]
// CHECK-NEXT:        ttl.tile_regs_release
// CHECK-NEXT:        ttl.yield %[[SQRT]] : !ttcore.tile<32x32, f32>
// CHECK-NEXT:      } -> tensor<2x2x!ttcore.tile<32x32, f32>>
// CHECK-NEXT:      return %[[RES]]
func.func @seven_op_chain(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                          %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %output = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>

  %a_ready = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_ready = ttl.cb_wait %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %output_cb = ttl.attach_cb %output, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_ready, %b_ready : tensor<2x2x!ttcore.tile<32x32, f32>>,
                               tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%output_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    // Seven-operation fused chain - each must appear in output with dst_idx
    %add = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %sub = ttl.tile_sub %add, %b_tile : !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %sub, %b_tile : !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %mul : !ttcore.tile<32x32, f32>
    %log = ttl.tile_log %exp : !ttcore.tile<32x32, f32>
    %neg = ttl.tile_neg %log : !ttcore.tile<32x32, f32>
    %sqrt = ttl.tile_sqrt %neg : !ttcore.tile<32x32, f32>
    %result_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %sqrt, %result_view : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield %sqrt : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
