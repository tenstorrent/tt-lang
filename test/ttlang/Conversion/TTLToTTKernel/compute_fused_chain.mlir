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
// FPU:       ttkernel.binary_op_init_common
// FPU:       scf.for
// FPU:         scf.for
// FPU:           ttkernel.tile_regs_acquire
// FPU:           ttkernel.add_tiles_init
// FPU:           ttkernel.add_tiles(
// FPU-NOT:       ttkernel.add_binary_tile
// mul's rhs needs copy_tile (from CB to DST)
// FPU:           ttkernel.copy_tile_init
// FPU:           ttkernel.copy_tile(
// FPU:           ttkernel.mul_binary_tile_init
// FPU:           ttkernel.mul_binary_tile(
// FPU:           ttkernel.exp_tile_init
// FPU:           ttkernel.exp_tile(
// FPU:           ttkernel.tile_regs_commit
// FPU:           ttkernel.tile_regs_wait
// FPU:           ttkernel.pack_tile(
// FPU:           ttkernel.tile_regs_release

// =============================================================================
// SFPU path checks
// =============================================================================
// All binary ops use copy_tile + SFPU binary.
// SFPU-LABEL: func.func @fused_chain_lowering
// SFPU:       ttkernel.init_sfpu
// SFPU:       scf.for
// SFPU:         scf.for
// SFPU:           ttkernel.tile_regs_acquire
// SFPU:           ttkernel.copy_tile_init
// SFPU:           ttkernel.copy_tile(
// SFPU:           ttkernel.copy_tile_init
// SFPU:           ttkernel.copy_tile(
// SFPU:           ttkernel.add_binary_tile_init
// SFPU:           ttkernel.add_binary_tile(
// SFPU:           ttkernel.mul_binary_tile_init
// SFPU:           ttkernel.mul_binary_tile(
// SFPU:           ttkernel.exp_tile_init
// SFPU:           ttkernel.exp_tile(
// SFPU:           ttkernel.tile_regs_commit
// SFPU:           ttkernel.tile_regs_wait
// SFPU:           ttkernel.pack_tile(
// SFPU:           ttkernel.tile_regs_release
// SFPU-NOT:       ttkernel.add_tiles

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
