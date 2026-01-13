// Summary: Fused add->mul->exp lowers through loops to TTKernel ops (with sync).
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-tile-and-assign-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, canonicalize, cse)' \
// RUN:   | FileCheck %s

// Purpose: ensure copy_tile + fused math ops lower to ttkernel with no TTL ops left.
// After conversion, attach_cb ops are removed (replaced with their tensor operands).
// CHECK-LABEL: func.func @fused_chain_lowering
// CHECK-SAME: (%[[AARG:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[BARG:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : i32
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[OUTPUT:.*]] = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
// CHECK:       %[[CB0_TTK:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK:       %[[CB1_TTK:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK:       %[[CB2_TTK:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK-NEXT:  ttkernel.cb_wait_front(%[[CB0_TTK]], %[[C4]])
// CHECK-NEXT:  ttkernel.cb_wait_front(%[[CB1_TTK]], %[[C4]])
// CHECK-NEXT:  ttkernel.init_sfpu(%[[CB0_TTK]], %[[CB2_TTK]])
// CHECK-NEXT:  ttkernel.tile_regs_acquire
// Compute loop (math ops only, no pack ops)
// CHECK-NEXT:  scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-NEXT:    scf.for %[[J:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
// Compute linear tile index: i * cols + j (via affine map)
// CHECK-NEXT:      %[[LINIDX:.*]] = affine.apply #{{.*}}(%[[I]], %[[J]])
// CHECK-NEXT:      ttkernel.copy_tile_init(%[[CB0_TTK]])
// CHECK-NEXT:      ttkernel.copy_tile(%[[CB0_TTK]], %[[LINIDX]], %[[C0]])
// CHECK-NEXT:      ttkernel.copy_tile_init(%[[CB1_TTK]])
// CHECK-NEXT:      ttkernel.copy_tile(%[[CB1_TTK]], %[[LINIDX]], %[[C1]])
// CHECK-NEXT:      ttkernel.add_binary_tile_init()
// CHECK-NEXT:      ttkernel.add_binary_tile(%[[C0]], %[[C1]], %[[C0]])
// CHECK-NEXT:      ttkernel.mul_binary_tile_init()
// CHECK-NEXT:      ttkernel.mul_binary_tile(%[[C0]], %[[C1]], %[[C0]])
// CHECK-NEXT:      ttkernel.exp_tile_init()
// CHECK-NEXT:      ttkernel.exp_tile(%[[C0]])
// End compute loops (no iter_args needed - results in DST registers)
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// Sync ops OUTSIDE compute loops (multitile fix)
// CHECK-NEXT:  ttkernel.tile_regs_commit
// CHECK-NEXT:  ttkernel.tile_regs_wait
// Pack loop (separate from compute loop, with iter_args for tensor updates)
// CHECK-NEXT:  %[[PACK_RESULT:.*]] = scf.for %[[PACK_I:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[PACK_ACC:.*]] = %[[OUTPUT]])
// CHECK-NEXT:    %[[INNER_RESULT:.*]] = scf.for %[[PACK_J:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[PACK_ACC2:.*]] = %[[PACK_ACC]])
// CHECK-NEXT:      %[[PACK_TILE:.*]] = tensor.extract %[[PACK_ACC2]][%[[PACK_I]], %[[PACK_J]]]
// CHECK-NEXT:      ttkernel.cb_reserve_back(%[[CB2_TTK]], %[[C4:.*]])
// Compute CB tile index: i * 2 + j (linearized row-major index for 2x2 grid).
// CHECK-NEXT:      %[[IOFF:.*]] = arith.muli %[[PACK_I]], %[[C2]] : index
// CHECK-NEXT:      %[[CB_IDX:.*]] = arith.addi %[[IOFF]], %[[PACK_J]] : index
// CHECK-NEXT:      ttkernel.pack_tile(%[[C0]], %[[CB2_TTK]], %[[CB_IDX]], false)
// CHECK-NEXT:      ttkernel.cb_push_back(%[[CB2_TTK]], %[[C4]])
// CHECK-NEXT:      %[[INSERT:.*]] = tensor.insert %[[PACK_TILE]] into %[[PACK_ACC2]][%[[PACK_I]], %[[PACK_J]]]
// CHECK-NEXT:      scf.yield %[[INSERT]]
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.yield %[[INNER_RESULT]]
// CHECK-NEXT:  }
// CHECK-NEXT:  ttkernel.tile_regs_release
// CHECK-NEXT:  return %[[PACK_RESULT]]
// CHECK-NOT:   ttl.attach_cb
// CHECK-NOT:   ttl.copy_tile
func.func @fused_chain_lowering(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %output = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>

  // Wait for input CBs (entire blocks) before compute.
  %a_ready = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_ready = ttl.cb_wait %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %output_cb = ttl.attach_cb %output, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_ready, %b_ready : tensor<2x2x!ttcore.tile<32x32, f32>>,
                               tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%output_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
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
    %result_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %exp, %result_view : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
