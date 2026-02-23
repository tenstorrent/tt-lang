// Summary: FPU binary init must derive CBs from the actual FPU binary op's
// operands, not from hardcoded compute input indices [0] and [1]. A 3-input
// compute where the FPU binary uses inputs 0 and 2 (skipping 1) verifies
// that binary_op_init_common references the correct CBs.
//
// The common init is now emitted by ttkernel-consolidate-inits (not the sync
// pass), so this test runs the full pipeline through conversion.

// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8}, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-consolidate-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// 3-input compute: tile_add(%a, %c) is FPU binary using inputs 0 and 2.
// Input 1 (%b) is consumed by an SFPU mul (one operand is computed).
// add_tiles_init must use cb0 and cb2 (FPU binary operand CBs), NOT cb0, cb1.

// CHECK-LABEL: func.func @init_binary_skips_middle_input
// CHECK-DAG: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// add_tiles_init must use cb0 and cb2 (the FPU binary operand CBs).
// CHECK: ttkernel.add_tiles_init(%[[CB0]], %[[CB2]])
// CHECK: ttkernel.add_tiles(%[[CB0]], %[[CB2]],
func.func @init_binary_skips_middle_input(
    %a: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %b: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %c: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb16 = ttl.bind_cb {cb_index = 16, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb16 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_cb, %b_cb, %c_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>,
       %c_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    // FPU binary: uses inputs 0 (a) and 2 (c), both block args -> FPU path
    %add = ttl.tile_add %a_tile, %c_tile : !ttcore.tile<32x32, f32>
    // SFPU binary: one operand is computed (add result), so b needs copy_tile
    %mul = ttl.tile_mul %add, %b_tile : !ttcore.tile<32x32, f32>
    ttl.yield %mul : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
