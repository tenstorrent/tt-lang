// Summary: Seven-operation fused chain to verify DST allocation handles long chains.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8},ttl-insert-tile-regs-sync))' | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8 separate-output-region=1},ttl-insert-tile-regs-sync))' | FileCheck %s --check-prefix=SEPARATE
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8 enable-fpu-binary-ops=0},ttl-insert-tile-regs-sync))' | FileCheck %s --check-prefix=SFPU

// Verify no placeholder copies remain in final IR
// CHECK-NOT: placeholder

// Purpose: Regression test for DST register allocation bug where operations were
// dropped in fused chains due to register conflicts. This test verifies that all
// seven operations in the chain receive dst_idx attributes and appear in output.
// The chain: add -> sub -> mul -> exp -> log -> neg -> sqrt should all be present.
// The initial add(a, b) is FPU binary because both operands are block args
// (directly from CB), so no copy_tile is needed for A or B.
// Only B needs a copy_tile for sub and mul which consume it as an SFPU operand.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL:   func.func @seven_op_chain
// CHECK:           %[[CB0:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 1}
// CHECK:           %[[CB1:.*]] = ttl.bind_cb{cb_index = 1, buffer_factor = 1}
// CHECK:           %[[CB2:.*]] = ttl.bind_cb{cb_index = 2, buffer_factor = 1}
// CHECK:           %[[RES:.*]] = ttl.compute
// CHECK:           ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[B:.*]]: !ttcore.tile<32x32, f32>, %[[O:.*]]: !ttcore.tile<32x32, f32>):
// CHECK:             ttl.tile_regs_acquire
// CHECK-NEXT:        %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:        %[[I1:.*]] = ttl.iter_index 1 : index
// FPU binary add: both operands are block args, no copy_tile needed
// CHECK-NEXT:        %[[ADD:.*]] = ttl.tile_add %[[A]], %[[B]] {dst_idx = 0 : i32, ttl.fpu_binary}
// Copy B for sub/mul (SFPU operand needs copy_tile)
// CHECK:             %{{.*}}, %[[DTILE:.*]] = ttl.copy_tile %[[B]][%[[I0]], %[[I1]]], %{{.*}} {dst_idx = 1 : i32}
// CHECK-NEXT:        %[[SUB:.*]] = ttl.tile_sub %[[ADD]], %[[DTILE]] {dst_idx = 0 : i32}
// CHECK-NEXT:        %[[MUL:.*]] = ttl.tile_mul %[[SUB]], %[[DTILE]] {dst_idx = 0 : i32}
// CHECK-NEXT:        %[[EXP:.*]] = ttl.tile_exp %[[MUL]] {dst_idx = 0 : i32}
// CHECK-NEXT:        %[[LOG:.*]] = ttl.tile_log %[[EXP]] {dst_idx = 0 : i32}
// CHECK-NEXT:        %[[NEG:.*]] = ttl.tile_neg %[[LOG]] {dst_idx = 0 : i32}
// CHECK-NEXT:        %[[SQRT:.*]] = ttl.tile_sqrt %[[NEG]] {dst_idx = 0 : i32}
// SEPARATE: ttl.tile_sqrt {{.*}} {dst_idx = 2 : i32}
//
// SFPU path: init_sfpu instead of init_binary, copy_tile for both add operands
// SFPU-LABEL:   func.func @seven_op_chain
// SFPU:           %[[CB0S:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 1}
// SFPU:           %[[CB2S:.*]] = ttl.bind_cb{cb_index = 2, buffer_factor = 1}
// SFPU:           ttl.compute
// SFPU:             ttl.tile_regs_acquire
// SFPU-NOT:         fpu_binary
// copy A and B for SFPU add
// SFPU:             ttl.copy_tile {{.*}} {dst_idx = 0 : i32}
// SFPU:             ttl.copy_tile {{.*}} {dst_idx = 1 : i32}
// SFPU:             ttl.tile_add {{.*}} {dst_idx = 0 : i32}
// sub, mul reuse the B copy (dst_tile_1 at slot 1)
// SFPU:             ttl.tile_sub {{.*}} {dst_idx = 0 : i32}
// SFPU:             ttl.tile_mul {{.*}} {dst_idx = 0 : i32}
// SFPU:             ttl.tile_exp {{.*}} {dst_idx = 0 : i32}
// SFPU:             ttl.tile_log {{.*}} {dst_idx = 0 : i32}
// SFPU:             ttl.tile_neg {{.*}} {dst_idx = 0 : i32}
// SFPU:             %[[SQRTS:.*]] = ttl.tile_sqrt {{.*}} {dst_idx = 0 : i32}
// SFPU:             ttl.tile_regs_commit
// SFPU-NEXT:        ttl.tile_regs_wait
// SFPU:             ttl.tile_store %[[SQRTS]], %{{.*}}[%{{.*}}, %{{.*}}]
// SFPU-NEXT:        ttl.tile_regs_release
// SFPU-NEXT:        ttl.yield
//
// CHECK-NEXT:        ttl.tile_regs_commit
// CHECK-NEXT:        ttl.tile_regs_wait
// CHECK-NEXT:        ttl.tile_store %[[SQRT]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:        ttl.tile_regs_release
// CHECK-NEXT:        ttl.yield
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

  %result_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_ready, %b_ready : tensor<2x2x!ttcore.tile<32x32, f32>>,
                               tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%output_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    // Seven-operation fused chain - each must appear in output with dst_idx
    %add = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %sub = ttl.tile_sub %add, %b_tile : !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %sub, %b_tile : !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %mul : !ttcore.tile<32x32, f32>
    %log = ttl.tile_log %exp : !ttcore.tile<32x32, f32>
    %neg = ttl.tile_neg %log : !ttcore.tile<32x32, f32>
    %sqrt = ttl.tile_sqrt %neg : !ttcore.tile<32x32, f32>
    ttl.tile_store %sqrt, %result_view[%i, %j] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
