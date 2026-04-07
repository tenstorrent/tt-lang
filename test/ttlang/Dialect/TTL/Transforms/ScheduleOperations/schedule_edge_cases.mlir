// Tests for scheduling edge cases: single-type (already sorted), single-tile,
// and multi-type chains that exercise different scheduler code paths.

// FPU path (default): binary add uses add_tiles (reads from CB).
// RUN: ttlang-opt %s --split-input-file \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst, ttl-subblock-compute-for-dst, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s --check-prefix=FPU

// SFPU path: binary add uses copy_tile + add_binary_tile.
// RUN: ttlang-opt %s --split-input-file \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-subblock-compute-for-dst, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s --check-prefix=SFPU

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 1: Single-type binary (already sorted path)
// =============================================================================
// Purpose: With only one compute op type, the scheduler's is_sorted check
// returns true and no reordering occurs.

// FPU-LABEL: func.func @single_type_already_sorted
// FPU-DAG:   %[[C2_I32:.*]] = arith.constant 2 : i32
// FPU-DAG:   %[[C1:.*]] = arith.constant 1 : index
// FPU-DAG:   %[[C0:.*]] = arith.constant 0 : index
// FPU:       %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// FPU:       %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// FPU:       %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// FPU:       ttkernel.cb_wait_front(%[[CB0]], %[[C2_I32]])
// FPU:       ttkernel.cb_wait_front(%[[CB1]], %[[C2_I32]])
// FPU:       ttkernel.cb_reserve_back(%[[CB2]], %[[C2_I32]])
// FPU:       ttkernel.binary_op_init_common(%[[CB0]], %[[CB1]], %[[CB2]])
// FPU:       ttkernel.tile_regs_acquire
// FPU:       ttkernel.add_tiles_init(%[[CB0]], %[[CB1]])
// FPU:       ttkernel.add_tiles(%[[CB0]], %[[CB1]], %[[C0]], %[[C0]], %[[C0]])
// FPU-NOT:   ttkernel.add_tiles_init
// FPU:       ttkernel.add_tiles(%[[CB0]], %[[CB1]], %[[C1]], %[[C1]], %[[C1]])
// FPU:       ttkernel.tile_regs_commit
// FPU-NEXT:  ttkernel.tile_regs_wait
// Pack phase: pack_tile after wait, cb_push_back after release
// FPU:       ttkernel.pack_tile(%[[C0]], %[[CB2]], %[[C0]], true)
// FPU:       ttkernel.pack_tile(%[[C1]], %[[CB2]], %[[C1]], true)
// FPU:       ttkernel.tile_regs_release
// FPU:       ttkernel.cb_push_back(%[[CB2]], %[[C2_I32]])
// FPU-NOT:   ttkernel.copy_tile
// FPU-NOT:   ttkernel.add_binary_tile

// SFPU-LABEL: func.func @single_type_already_sorted
// SFPU-DAG:  %[[C3:.*]] = arith.constant 3 : index
// SFPU-DAG:  %[[C2_I32:.*]] = arith.constant 2 : i32
// SFPU-DAG:  %[[C2:.*]] = arith.constant 2 : index
// SFPU-DAG:  %[[C1:.*]] = arith.constant 1 : index
// SFPU-DAG:  %[[C0:.*]] = arith.constant 0 : index
// SFPU:      %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// SFPU:      %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// SFPU:      %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// SFPU:      ttkernel.cb_wait_front(%[[CB0]], %[[C2_I32]])
// SFPU:      ttkernel.cb_wait_front(%[[CB1]], %[[C2_I32]])
// SFPU:      ttkernel.cb_reserve_back(%[[CB2]], %[[C2_I32]])
// SFPU:      ttkernel.init_sfpu(%[[CB0]], %[[CB2]])
// SFPU:      ttkernel.tile_regs_acquire
// Grouped: all copies from CB0, all copies from CB1, then all adds
// SFPU:      ttkernel.copy_tile_init(%[[CB0]])
// SFPU:      ttkernel.copy_tile(%[[CB0]], %[[C0]], %[[C0]])
// SFPU:      ttkernel.copy_tile(%[[CB0]], %[[C1]], %[[C2]])
// SFPU:      ttkernel.copy_tile_init(%[[CB1]])
// SFPU:      ttkernel.copy_tile(%[[CB1]], %[[C0]], %[[C1]])
// SFPU:      ttkernel.copy_tile(%[[CB1]], %[[C1]], %[[C3]])
// SFPU:      ttkernel.add_binary_tile_init
// SFPU:      ttkernel.add_binary_tile(%[[C0]], %[[C1]], %[[C0]])
// SFPU:      ttkernel.add_binary_tile(%[[C2]], %[[C3]], %[[C2]])
// SFPU:      ttkernel.tile_regs_commit
// SFPU-NEXT: ttkernel.tile_regs_wait
// Pack phase: pack_tile after wait, cb_push_back after release
// SFPU:      ttkernel.pack_tile(%[[C0]], %[[CB2]], %[[C0]], true)
// SFPU:      ttkernel.pack_tile(%[[C2]], %[[CB2]], %[[C1]], true)
// SFPU:      ttkernel.tile_regs_release
// SFPU:      ttkernel.cb_push_back(%[[CB2]], %[[C2_I32]])
// SFPU-NOT:  ttkernel.add_tiles

func.func @single_type_already_sorted(
    %a: tensor<2x1x!ttcore.tile<32x32, bf16>>,
    %b: tensor<2x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<2x1x!ttcore.tile<32x32, bf16>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %output = tensor.empty() : tensor<2x1x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>

  %a_ready = ttl.cb_wait %cb0 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %b_ready = ttl.cb_wait %cb1 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %output_cb = ttl.attach_cb %output, %cb2 : (tensor<2x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>

  %result_view = ttl.cb_reserve %cb2 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>

  %result = ttl.compute
      ins(%a_ready, %b_ready : tensor<2x1x!ttcore.tile<32x32, bf16>>,
                               tensor<2x1x!ttcore.tile<32x32, bf16>>)
      outs(%output_cb : tensor<2x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, bf16>,
       %b_tile: !ttcore.tile<32x32, bf16>,
       %out_tile: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, bf16>
    ttl.tile_store %sum, %result_view[%i, %j] : !ttcore.tile<32x32, bf16>, tensor<2x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<2x1x!ttcore.tile<32x32, bf16>>

  ttl.cb_push %cb2 : <[2, 1], !ttcore.tile<32x32, bf16>, 1>

  func.return %result : tensor<2x1x!ttcore.tile<32x32, bf16>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 2: Three-type chain: copy + binary + unary
// =============================================================================
// Purpose: Exercises reordering across scheduling categories in a single
// sync region. The add(a,b) and exp(c) are independent.
// FPU-LABEL: func.func @copy_before_sfpu_reorder
// FPU-DAG:   %[[C3:.*]] = arith.constant 3 : index
// FPU-DAG:   %[[C2_I32:.*]] = arith.constant 2 : i32
// FPU-DAG:   %[[C2:.*]] = arith.constant 2 : index
// FPU-DAG:   %[[C0:.*]] = arith.constant 0 : index
// FPU-DAG:   %[[C1:.*]] = arith.constant 1 : index
// FPU:       %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// FPU:       %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// FPU:       %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// FPU:       %[[CB3:.*]] = ttkernel.get_compile_time_arg_val(3)
// FPU:       %[[CB4:.*]] = ttkernel.get_compile_time_arg_val(4)
// FPU:       ttkernel.cb_wait_front(%[[CB0]], %[[C2_I32]])
// FPU:       ttkernel.cb_wait_front(%[[CB1]], %[[C2_I32]])
// FPU:       ttkernel.cb_wait_front(%[[CB2]], %[[C2_I32]])
// FPU:       ttkernel.cb_reserve_back(%[[CB3]], %[[C2_I32]])
// FPU:       ttkernel.cb_reserve_back(%[[CB4]], %[[C2_I32]])
// FPU:       ttkernel.binary_op_init_common(%[[CB0]], %[[CB1]], %[[CB3]])
// FPU:       ttkernel.tile_regs_acquire
// Grouped: copy(c) for both tiles, add_tiles for both tiles, exp for both tiles
// FPU:       ttkernel.copy_tile_init(%[[CB2]])
// FPU:       ttkernel.copy_tile(%[[CB2]], %[[C0]], %[[C1]])
// FPU:       ttkernel.copy_tile(%[[CB2]], %[[C1]], %[[C3]])
// FPU:       ttkernel.add_tiles_init(%[[CB0]], %[[CB1]])
// FPU:       ttkernel.add_tiles(%[[CB0]], %[[CB1]], %[[C0]], %[[C0]], %[[C0]])
// FPU:       ttkernel.add_tiles(%[[CB0]], %[[CB1]], %[[C1]], %[[C1]], %[[C2]])
// FPU:       ttkernel.exp_tile_init
// FPU:       ttkernel.exp_tile(%[[C1]])
// FPU:       ttkernel.exp_tile(%[[C3]])
// FPU:       ttkernel.tile_regs_commit
// FPU-NEXT:  ttkernel.tile_regs_wait
// Pack phase
// FPU:       ttkernel.pack_tile(%[[C0]], %[[CB3]], %[[C0]], true)
// FPU:       ttkernel.pack_tile(%[[C1]], %[[CB4]], %[[C0]], true)
// FPU:       ttkernel.pack_tile(%[[C2]], %[[CB3]], %[[C1]], true)
// FPU:       ttkernel.pack_tile(%[[C3]], %[[CB4]], %[[C1]], true)
// FPU:       ttkernel.tile_regs_release
// FPU:       ttkernel.cb_push_back(%[[CB3]], %[[C2_I32]])
// FPU:       ttkernel.cb_push_back(%[[CB4]], %[[C2_I32]])
// FPU-NOT:   ttkernel.add_binary_tile

// SFPU-LABEL: func.func @copy_before_sfpu_reorder
// SFPU-DAG:  %[[C3:.*]] = arith.constant 3 : index
// SFPU-DAG:  %[[C2_I32:.*]] = arith.constant 2 : i32
// SFPU-DAG:  %[[C2:.*]] = arith.constant 2 : index
// SFPU-DAG:  %[[C1:.*]] = arith.constant 1 : index
// SFPU-DAG:  %[[C0:.*]] = arith.constant 0 : index
// SFPU:      %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// SFPU:      %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// SFPU:      %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// SFPU:      %[[CB3:.*]] = ttkernel.get_compile_time_arg_val(3)
// SFPU:      %[[CB4:.*]] = ttkernel.get_compile_time_arg_val(4)
// SFPU:      ttkernel.cb_wait_front(%[[CB0]], %[[C2_I32]])
// SFPU:      ttkernel.cb_wait_front(%[[CB1]], %[[C2_I32]])
// SFPU:      ttkernel.cb_wait_front(%[[CB2]], %[[C2_I32]])
// SFPU:      ttkernel.cb_reserve_back(%[[CB3]], %[[C2_I32]])
// SFPU:      ttkernel.cb_reserve_back(%[[CB4]], %[[C2_I32]])
// SFPU:      ttkernel.init_sfpu(%[[CB0]], %[[CB3]])
// SFPU:      ttkernel.tile_regs_acquire
// Grouped: copies from CB0, copies from CB1, adds, copies from CB2, exps
// SFPU:      ttkernel.copy_tile_init(%[[CB0]])
// SFPU:      ttkernel.copy_tile(%[[CB0]], %[[C0]], %[[C0]])
// SFPU:      ttkernel.copy_tile(%[[CB0]], %[[C1]], %[[C2]])
// SFPU:      ttkernel.copy_tile_init(%[[CB1]])
// SFPU:      ttkernel.copy_tile(%[[CB1]], %[[C0]], %[[C1]])
// SFPU:      ttkernel.copy_tile(%[[CB1]], %[[C1]], %[[C3]])
// SFPU:      ttkernel.add_binary_tile_init
// SFPU:      ttkernel.add_binary_tile(%[[C0]], %[[C1]], %[[C0]])
// SFPU:      ttkernel.add_binary_tile(%[[C2]], %[[C3]], %[[C2]])
// SFPU:      ttkernel.copy_tile_init(%[[CB2]])
// SFPU:      ttkernel.copy_tile(%[[CB2]], %[[C0]], %[[C1]])
// SFPU:      ttkernel.copy_tile(%[[CB2]], %[[C1]], %[[C3]])
// SFPU:      ttkernel.exp_tile_init
// SFPU:      ttkernel.exp_tile(%[[C1]])
// SFPU:      ttkernel.exp_tile(%[[C3]])
// SFPU:      ttkernel.tile_regs_commit
// SFPU-NEXT: ttkernel.tile_regs_wait
// Pack phase
// SFPU:      ttkernel.pack_tile(%[[C0]], %[[CB3]], %[[C0]], true)
// SFPU:      ttkernel.pack_tile(%[[C1]], %[[CB4]], %[[C0]], true)
// SFPU:      ttkernel.pack_tile(%[[C2]], %[[CB3]], %[[C1]], true)
// SFPU:      ttkernel.pack_tile(%[[C3]], %[[CB4]], %[[C1]], true)
// SFPU:      ttkernel.tile_regs_release
// SFPU:      ttkernel.cb_push_back(%[[CB3]], %[[C2_I32]])
// SFPU:      ttkernel.cb_push_back(%[[CB4]], %[[C2_I32]])
// SFPU-NOT:  ttkernel.add_tiles

func.func @copy_before_sfpu_reorder(
    %a: tensor<2x1x!ttcore.tile<32x32, bf16>>,
    %b: tensor<2x1x!ttcore.tile<32x32, bf16>>,
    %c: tensor<2x1x!ttcore.tile<32x32, bf16>>)
    -> (tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %out0 = tensor.empty() : tensor<2x1x!ttcore.tile<32x32, bf16>>
  %out1 = tensor.empty() : tensor<2x1x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb4 = ttl.bind_cb {cb_index = 4, block_count = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>

  %a_ready = ttl.cb_wait %cb0 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %b_ready = ttl.cb_wait %cb1 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %c_ready = ttl.cb_wait %cb2 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %out0_cb = ttl.attach_cb %out0, %cb3 : (tensor<2x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %out1_cb = ttl.attach_cb %out1, %cb4 : (tensor<2x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>

  %rv0 = ttl.cb_reserve %cb3 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %rv1 = ttl.cb_reserve %cb4 : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>

  %result:2 = ttl.compute
      ins(%a_ready, %b_ready, %c_ready : tensor<2x1x!ttcore.tile<32x32, bf16>>,
                                          tensor<2x1x!ttcore.tile<32x32, bf16>>,
                                          tensor<2x1x!ttcore.tile<32x32, bf16>>)
      outs(%out0_cb, %out1_cb : tensor<2x1x!ttcore.tile<32x32, bf16>>,
                                tensor<2x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map, #map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, bf16>,
       %b_tile: !ttcore.tile<32x32, bf16>,
       %c_tile: !ttcore.tile<32x32, bf16>,
       %o0: !ttcore.tile<32x32, bf16>,
       %o1: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    // Binary at depth 0: reads from CB
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, bf16>
    // Unary: exp of c (needs copy_tile, independent of add)
    %exp = ttl.tile_exp %c_tile : !ttcore.tile<32x32, bf16>
    ttl.tile_store %sum, %rv0[%i, %j] : !ttcore.tile<32x32, bf16>, tensor<2x1x!ttcore.tile<32x32, bf16>>
    ttl.tile_store %exp, %rv1[%i, %j] : !ttcore.tile<32x32, bf16>, tensor<2x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> (tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>)

  ttl.cb_push %cb3 : <[2, 1], !ttcore.tile<32x32, bf16>, 1>
  ttl.cb_push %cb4 : <[2, 1], !ttcore.tile<32x32, bf16>, 1>

  func.return %result#0, %result#1 : tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>
}
