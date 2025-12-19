// Summary: Fused add->mul->exp lowers through loops to TTKernel ops (with sync).
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-tile-and-assign-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops), convert-ttl-to-ttkernel, canonicalize, cse)' \
// RUN:   | FileCheck %s

// Purpose: ensure copy_tile + fused math ops lower to ttkernel with no TTL ops left.
// CHECK-LABEL: func.func @fused_chain_lowering
// CHECK-SAME: (%[[AARG:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[BARG:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>)
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[INIT:.*]] = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
// CHECK:       %[[CB0_TTK:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK:       %[[CB1_TTK:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK:       %[[CB2_TTK:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK:       %[[A_CB:.*]] = ttl.attach_cb %[[AARG]],
// CHECK:       %[[B_CB:.*]] = ttl.attach_cb %[[BARG]],
// CHECK:       %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT]],
// CHECK:       scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[INIT_CB]])
// CHECK:         scf.for %[[J:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ACC2:.*]] = %[[ACC]])
// CHECK:           %[[ATILE:.*]] = tensor.extract %[[A_CB]][%[[I]], %[[J]]]
// CHECK:           ttkernel.tile_regs_acquire
// CHECK:           ttkernel.copy_tile_init(%[[CB0_TTK]])
// CHECK:           ttkernel.copy_tile(%[[CB0_TTK]], %[[IDX0:.*]], %[[DST0:.*]])
// CHECK:           ttkernel.copy_tile_init(%[[CB1_TTK]])
// CHECK:           ttkernel.copy_tile(%[[CB1_TTK]], %[[IDX0]], %[[DST1:.*]])
// CHECK:           ttkernel.add_binary_tile_init()
// CHECK:           ttkernel.add_binary_tile(%[[DST0]], %[[DST1]], %[[DST2:.*]])
// CHECK:           ttkernel.mul_binary_tile_init()
// CHECK:           ttkernel.mul_binary_tile(%[[DST2]], %[[DST1]], %[[DST0]])
// CHECK:           ttkernel.exp_tile_init()
// CHECK:           ttkernel.exp_tile(%[[DST1]])
// CHECK:           ttkernel.tile_regs_commit
// CHECK:           %[[INSERT:.*]] = tensor.insert %[[ATILE]] into %[[ACC2]][%[[I]], %[[J]]]
// CHECK:         scf.yield %[[INSERT]]
// CHECK:       scf.yield
// CHECK:       ttkernel.tile_regs_wait
// CHECK:       ttkernel.tile_regs_release
// CHECK:       return
// CHECK-NOT:   ttl.copy_tile %[[INIT_CB]]
func.func @fused_chain_lowering(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                         tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %sum, %b_tile : !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %mul : !ttcore.tile<32x32, f32>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
