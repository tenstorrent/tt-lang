// Summary: Fused add->mul->exp lowers through loops to TTKernel ops (with sync).
// Tests both FPU binary (default) and SFPU binary (disabled) paths.

// FPU path (default): add uses add_tiles (reads from CB), mul uses SFPU (mixed inputs).
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s --check-prefix=FPU

// SFPU path: all binary ops use copy_tile + SFPU binary ops.
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s --check-prefix=SFPU

// =============================================================================
// FPU path checks
// =============================================================================
// FPU binary add reads from CBs (no copy_tile for add operands).
// mul is SFPU because lhs is intermediate result in DST, needs copy_tile for rhs.
// FPU-LABEL: func.func @fused_chain_lowering
// FPU-SAME: (%[[AARG:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[BARG:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>)
// FPU-DAG:   %[[C4:.*]] = arith.constant 4 : i32
// FPU-DAG:   %[[C2:.*]] = arith.constant 2 : index
// FPU-DAG:   %[[C1:.*]] = arith.constant 1 : index
// FPU-DAG:   %[[C0:.*]] = arith.constant 0 : index
// FPU:       %[[OUTPUT:.*]] = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
// FPU:       %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// FPU:       %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// FPU:       %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// FPU:       ttkernel.cb_wait_front(%[[CB0]], %[[C4]])
// FPU:       ttkernel.cb_wait_front(%[[CB1]], %[[C4]])
// FPU:       ttkernel.cb_reserve_back(%[[CB2]], %[[C4]])
// FPU:       ttkernel.binary_op_init_common(%[[CB0]], %[[CB1]], %[[CB2]])
// FPU:       scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// FPU:         scf.for %[[J:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// FPU:           %[[AFFINEIDX:.*]] = affine.apply #{{.*}}(%[[I]], %[[J]])
// FPU:           ttkernel.tile_regs_acquire
// Linearized CB index: i * 2 + j
// FPU:           %[[MULI:.*]] = arith.muli %[[I]], %[[C2]]
// FPU:           %[[LINIDX:.*]] = arith.addi %[[MULI]], %[[J]]
// FPU:           ttkernel.add_tiles_init(%[[CB0]], %[[CB1]])
// add_tiles reads lhs and rhs from CB at linearized index, writes DST[0]
// FPU:           ttkernel.add_tiles(%[[CB0]], %[[CB1]], %[[LINIDX]], %[[LINIDX]], %[[C0]])
// FPU-NOT:       ttkernel.add_binary_tile
// mul's rhs needs copy_tile (from CB1 to DST[1])
// FPU:           ttkernel.copy_tile_init(%[[CB1]])
// FPU:           ttkernel.copy_tile(%[[CB1]], %[[AFFINEIDX]], %[[C1]])
// FPU:           ttkernel.mul_binary_tile_init
// FPU:           ttkernel.mul_binary_tile(%[[C0]], %[[C1]], %[[C0]])
// FPU:           ttkernel.exp_tile_init
// FPU:           ttkernel.exp_tile(%[[C0]])
// FPU:           ttkernel.tile_regs_commit
// FPU:           ttkernel.tile_regs_wait
// FPU:           ttkernel.pack_tile(%[[C0]], %[[CB2]], %[[LINIDX]], true)
// FPU:           ttkernel.cb_push_back(%[[CB2]], %[[C4]])
// FPU:           ttkernel.tile_regs_release
// FPU:         }
// FPU:       }
// FPU:       return
// FPU-NOT:   ttl.attach_cb
// FPU-NOT:   ttl.copy_tile

// =============================================================================
// SFPU path checks
// =============================================================================
// All binary ops use copy_tile + SFPU binary.
// SFPU-LABEL: func.func @fused_chain_lowering
// SFPU-SAME: (%[[AARG:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[BARG:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>)
// SFPU-DAG:   %[[C4:.*]] = arith.constant 4 : i32
// SFPU-DAG:   %[[C2:.*]] = arith.constant 2 : index
// SFPU-DAG:   %[[C1:.*]] = arith.constant 1 : index
// SFPU-DAG:   %[[C0:.*]] = arith.constant 0 : index
// SFPU:       %[[OUTPUT:.*]] = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
// SFPU:       %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// SFPU:       %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// SFPU:       %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// SFPU:       ttkernel.cb_wait_front(%[[CB0]], %[[C4]])
// SFPU:       ttkernel.cb_wait_front(%[[CB1]], %[[C4]])
// SFPU:       ttkernel.cb_reserve_back(%[[CB2]], %[[C4]])
// SFPU:       ttkernel.init_sfpu(%[[CB0]], %[[CB2]])
// SFPU:       scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// SFPU:         scf.for %[[J:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// SFPU:           %[[LINIDX:.*]] = affine.apply #{{.*}}(%[[I]], %[[J]])
// SFPU:           ttkernel.tile_regs_acquire
// Copies at first use (add): CB0 first, then CB1
// SFPU:           ttkernel.copy_tile_init(%[[CB0]])
// SFPU:           ttkernel.copy_tile(%[[CB0]], %[[LINIDX]], %[[C0]])
// SFPU:           ttkernel.copy_tile_init(%[[CB1]])
// SFPU:           ttkernel.copy_tile(%[[CB1]], %[[LINIDX]], %[[C1]])
// SFPU:           ttkernel.add_binary_tile_init()
// SFPU:           ttkernel.add_binary_tile(%[[C0]], %[[C1]], %[[C0]])
// SFPU:           ttkernel.mul_binary_tile_init()
// SFPU:           ttkernel.mul_binary_tile(%[[C0]], %[[C1]], %[[C0]])
// SFPU:           ttkernel.exp_tile_init()
// SFPU:           ttkernel.exp_tile(%[[C0]])
// SFPU:           ttkernel.tile_regs_commit
// SFPU:           ttkernel.tile_regs_wait
// Linearized CB index for pack: i * 2 + j
// SFPU:           %[[SMULI:.*]] = arith.muli %[[I]], %[[C2]]
// SFPU:           %[[SLINIDX:.*]] = arith.addi %[[SMULI]], %[[J]]
// SFPU:           ttkernel.pack_tile(%[[C0]], %[[CB2]], %[[SLINIDX]], true)
// SFPU:           ttkernel.cb_push_back(%[[CB2]], %[[C4]])
// SFPU:           ttkernel.tile_regs_release
// SFPU:         }
// SFPU:       }
// SFPU:       return
// SFPU-NOT:   ttl.attach_cb
// SFPU-NOT:   ttl.copy_tile
// SFPU-NOT:   ttkernel.add_tiles

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

  %result_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
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
    ttl.tile_store %exp, %result_view : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
