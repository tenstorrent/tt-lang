// Summary: Fused add->mul->exp lowers through loops to TTKernel ops (with sync).
// Tests both FPU binary (default) and SFPU binary (disabled) paths.

// FPU path (default): add uses add_tiles (reads from CB), mul uses SFPU (mixed inputs).
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s --check-prefix=FPU

// SFPU path: all binary ops use copy_tile + SFPU binary ops.
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
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
// FPU:           ttkernel.tile_regs_acquire
// FPU:           %[[LINIDX:.*]] = affine.linearize_index [%[[I]], %[[J]]] by (2, 2)
// FPU:           ttkernel.add_tiles_init(%[[CB0]], %[[CB1]])
// FPU:           ttkernel.add_tiles(%[[CB0]], %[[CB1]], %[[LINIDX]], %[[LINIDX]], %[[C0]])
// FPU-NOT:       ttkernel.add_binary_tile
// FPU:           ttkernel.copy_tile_init(%[[CB1]])
// FPU:           ttkernel.copy_tile(%[[CB1]], %[[LINIDX]], %[[C1]])
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
// SFPU:           ttkernel.tile_regs_acquire
// Linearized CB index via affine.linearize_index
// SFPU:           %[[LINIDX:.*]] = affine.linearize_index [%[[I]], %[[J]]] by (2, 2)
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
// pack_tile uses same linearized index
// SFPU:           ttkernel.pack_tile(%[[C0]], %[[CB2]], %[[LINIDX]], true)
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

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>

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
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %c0 = arith.constant 0 : index
    %sum = ttl.tile_add %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %sum, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %mul into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %result_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
