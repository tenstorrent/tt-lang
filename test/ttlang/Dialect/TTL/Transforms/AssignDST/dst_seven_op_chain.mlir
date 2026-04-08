// Summary: Seven-operation fused chain to verify DST allocation handles long chains.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8}))' | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8 separate-output-region=1}))' | FileCheck %s --check-prefix=SEPARATE
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8 enable-fpu-binary-ops=0}))' | FileCheck %s --check-prefix=SFPU

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
// CHECK:           %[[CB0:.*]] = ttl.bind_cb{cb_index = 0, block_count = 1}
// CHECK:           %[[CB1:.*]] = ttl.bind_cb{cb_index = 1, block_count = 1}
// CHECK:           %[[CB2:.*]] = ttl.bind_cb{cb_index = 2, block_count = 1}
// CHECK:           %[[RES:.*]] = ttl.compute
// CHECK:           ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[B:.*]]: !ttcore.tile<32x32, f32>, %[[O:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:        %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:        %[[I1:.*]] = ttl.iter_index 1 : index
// FPU binary add: both operands are block args, no copy_tile needed
// CHECK:        %[[ADD:.*]] = ttl.tile_add %[[A]], %[[B]] into dst[{{.*}}] {ttl.fpu_binary}
// Copy B for sub/mul (SFPU operand needs copy_tile)
// CHECK:             %{{.*}}, %[[DTILE:.*]] = ttl.copy_tile %[[B]][%[[I0]], %[[I1]]], %
// CHECK:        %[[SUB:.*]] = ttl.tile_sub %[[ADD]], %[[DTILE]] into dst[{{.*}}]
// CHECK:        %[[MUL:.*]] = ttl.tile_mul %[[SUB]], %[[DTILE]] into dst[{{.*}}]
// CHECK:        %[[EXP:.*]] = ttl.tile_exp %[[MUL]] into dst[{{.*}}]
// CHECK:        %[[LOG:.*]] = ttl.tile_log %[[EXP]] into dst[{{.*}}]
// CHECK:        %[[NEG:.*]] = ttl.tile_neg %[[LOG]] into dst[{{.*}}]
// CHECK:        %[[SQRT:.*]] = ttl.tile_sqrt %[[NEG]] into dst[{{.*}}]
// SEPARATE: ttl.tile_sqrt {{.*}} into dst[{{.*}}]
//
// SFPU path: init_sfpu instead of init_binary, copy_tile for both add operands
// SFPU-LABEL:   func.func @seven_op_chain
// SFPU:           %[[CB0S:.*]] = ttl.bind_cb{cb_index = 0, block_count = 1}
// SFPU:           %[[CB2S:.*]] = ttl.bind_cb{cb_index = 2, block_count = 1}
// SFPU:           ttl.compute
// SFPU:           ^bb0
// SFPU-NOT:         fpu_binary
// copy A and B for SFPU add
// SFPU:             ttl.copy_tile {{.*}}
// SFPU:             ttl.copy_tile {{.*}}
// SFPU:             ttl.tile_add {{.*}} into dst[{{.*}}]
// sub, mul reuse the B copy (dst_tile_1 at slot 1)
// SFPU:             ttl.tile_sub {{.*}} into dst[{{.*}}]
// SFPU:             ttl.tile_mul {{.*}} into dst[{{.*}}]
// SFPU:             ttl.tile_exp {{.*}} into dst[{{.*}}]
// SFPU:             ttl.tile_log {{.*}} into dst[{{.*}}]
// SFPU:             ttl.tile_neg {{.*}} into dst[{{.*}}]
// SFPU:             %[[SQRTS:.*]] = ttl.tile_sqrt {{.*}} into dst[{{.*}}]
// SFPU:             ttl.tile_store %[[SQRTS]], %{{.*}}[%{{.*}}, %{{.*}}]{{.*}}into dst[{{.*}}]
// SFPU-NEXT:        ttl.yield
//
// CHECK-NEXT:        ttl.tile_store %[[SQRT]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:        ttl.yield
// CHECK-NEXT:      } -> tensor<2x2x!ttcore.tile<32x32, f32>>
// CHECK-NEXT:      return %[[RES]]
func.func @seven_op_chain(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                          %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %output = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>

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
    %c0 = arith.constant 0 : index
    %add = ttl.tile_add %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %sub = ttl.tile_sub %add, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %sub, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %mul into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %log = ttl.tile_log %exp into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %neg = ttl.tile_neg %log into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %sqrt = ttl.tile_sqrt %neg into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %sqrt, %result_view[%i, %j] into dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
