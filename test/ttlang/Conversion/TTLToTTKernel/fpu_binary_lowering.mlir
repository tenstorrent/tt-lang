// Summary: End-to-end test for FPU binary lowering through the full pipeline.
// When enable-fpu-binary-ops is enabled (default), binary add/sub/mul with
// both operands from CBs lower to FPU ops (ttkernel.add_tiles) instead of
// SFPU ops (ttkernel.add_binary_tile). FPU ops read from CBs, not DST.

// FPU path (default): add_tiles reads from CB, binary_op_init_common init.
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst, ttl-subblock-compute-for-dst{subblock-sync=true}, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   --split-input-file | FileCheck %s --check-prefix=FPU

// SFPU path: add_binary_tile reads from DST, init_sfpu init.
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-subblock-compute-for-dst{subblock-sync=true}, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   --split-input-file | FileCheck %s --check-prefix=SFPU

// =============================================================================
// Test 1: Simple FPU binary add (2x2 bf16)
// =============================================================================
// FPU path: dstPerIteration=1 (no copy_tile), unroll_factor=min(8,4)=4.
// All 4 tiles fit in one subblock, no outer loop.
// Expect: binary_op_init_common + 4x add_tiles + 4x pack_tile.
//
// SFPU path: dstPerIteration=2 (copy lhs + copy rhs), unroll_factor=min(4,4)=4.
// All 4 tiles fit, but each needs 2 copy_tiles.
// Expect: init_sfpu + 4x copy_tile(lhs) + 4x copy_tile(rhs) +
//         4x add_binary_tile + 4x pack_tile.

// FPU-LABEL: func.func @fpu_add_2x2
// FPU-DAG: %[[C3:.*]] = arith.constant 3 : index
// FPU-DAG: %[[C4I:.*]] = arith.constant 4 : i32
// FPU-DAG: %[[C2:.*]] = arith.constant 2 : index
// FPU-DAG: %[[C1:.*]] = arith.constant 1 : index
// FPU-DAG: %[[C0:.*]] = arith.constant 0 : index
// FPU:     %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// FPU:     %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// FPU:     %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// FPU:     ttkernel.cb_wait_front(%[[CB0]], %[[C4I]])
// FPU:     ttkernel.cb_wait_front(%[[CB2]], %[[C4I]])
// FPU:     ttkernel.cb_reserve_back(%[[CB1]], %[[C4I]])
// FPU:     ttkernel.binary_op_init_common(%[[CB0]], %[[CB2]], %[[CB1]])
// FPU:     ttkernel.tile_regs_acquire
// No copy_tile ops for FPU binary (operands read from CB)
// FPU-NOT: ttkernel.copy_tile
// FPU:     ttkernel.add_tiles_init(%[[CB0]], %[[CB2]])
// FPU:     ttkernel.add_tiles(%[[CB0]], %[[CB2]], %[[C0]], %[[C0]], %[[C0]])
// FPU:     ttkernel.add_tiles(%[[CB0]], %[[CB2]], %[[C1]], %[[C1]], %[[C1]])
// FPU:     ttkernel.add_tiles(%[[CB0]], %[[CB2]], %[[C2]], %[[C2]], %[[C2]])
// FPU:     ttkernel.add_tiles(%[[CB0]], %[[CB2]], %[[C3]], %[[C3]], %[[C3]])
// FPU:     ttkernel.tile_regs_commit
// FPU:     ttkernel.tile_regs_wait
// FPU:     ttkernel.pack_tile(%[[C0]], %[[CB1]], %[[C0]], true)
// FPU:     ttkernel.pack_tile(%[[C1]], %[[CB1]], %[[C1]], true)
// FPU:     ttkernel.pack_tile(%[[C2]], %[[CB1]], %[[C2]], true)
// FPU:     ttkernel.pack_tile(%[[C3]], %[[CB1]], %[[C3]], true)
// FPU:     ttkernel.tile_regs_release
// FPU:     ttkernel.cb_push_back(%[[CB1]], %[[C4I]])
// FPU-NOT: ttkernel.add_binary_tile

// SFPU-LABEL: func.func @fpu_add_2x2
// SFPU-DAG: %[[C7:.*]] = arith.constant 7 : index
// SFPU-DAG: %[[C5:.*]] = arith.constant 5 : index
// SFPU-DAG: %[[C3:.*]] = arith.constant 3 : index
// SFPU-DAG: %[[C6:.*]] = arith.constant 6 : index
// SFPU-DAG: %[[C4:.*]] = arith.constant 4 : index
// SFPU-DAG: %[[C4I:.*]] = arith.constant 4 : i32
// SFPU-DAG: %[[C2:.*]] = arith.constant 2 : index
// SFPU-DAG: %[[C1:.*]] = arith.constant 1 : index
// SFPU-DAG: %[[C0:.*]] = arith.constant 0 : index
// SFPU:     %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// SFPU:     %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// SFPU:     %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// SFPU:     ttkernel.cb_wait_front(%[[CB0]], %[[C4I]])
// SFPU:     ttkernel.cb_wait_front(%[[CB2]], %[[C4I]])
// SFPU:     ttkernel.cb_reserve_back(%[[CB1]], %[[C4I]])
// SFPU:     ttkernel.init_sfpu(%[[CB0]], %[[CB1]])
// SFPU:     ttkernel.tile_regs_acquire
// Grouped: all copies from CB0, all copies from CB2, then all adds
// SFPU:     ttkernel.copy_tile_init(%[[CB0]])
// SFPU:     ttkernel.copy_tile(%[[CB0]], %[[C0]], %[[C0]])
// SFPU:     ttkernel.copy_tile(%[[CB0]], %[[C1]], %[[C2]])
// SFPU:     ttkernel.copy_tile(%[[CB0]], %[[C2]], %[[C4]])
// SFPU:     ttkernel.copy_tile(%[[CB0]], %[[C3]], %[[C6]])
// SFPU:     ttkernel.copy_tile_init(%[[CB2]])
// SFPU:     ttkernel.copy_tile(%[[CB2]], %[[C0]], %[[C1]])
// SFPU:     ttkernel.copy_tile(%[[CB2]], %[[C1]], %[[C3]])
// SFPU:     ttkernel.copy_tile(%[[CB2]], %[[C2]], %[[C5]])
// SFPU:     ttkernel.copy_tile(%[[CB2]], %[[C3]], %[[C7]])
// SFPU:     ttkernel.add_binary_tile_init
// SFPU:     ttkernel.add_binary_tile(%[[C0]], %[[C1]], %[[C0]])
// SFPU:     ttkernel.add_binary_tile(%[[C2]], %[[C3]], %[[C2]])
// SFPU:     ttkernel.add_binary_tile(%[[C4]], %[[C5]], %[[C4]])
// SFPU:     ttkernel.add_binary_tile(%[[C6]], %[[C7]], %[[C6]])
// SFPU:     ttkernel.tile_regs_commit
// SFPU-NOT: ttkernel.add_tiles

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @fpu_add_2x2()
    attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %lhs_ready = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %lhs = ttl.attach_cb %lhs_ready, %cb0 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %rhs_ready = ttl.cb_wait %cb2 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %rhs = ttl.attach_cb %rhs_ready, %cb2 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %out_view = ttl.cb_reserve %cb1 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %out = ttl.attach_cb %out_view, %cb1 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %empty = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
  %out_cb = ttl.attach_cb %empty, %cb1 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute
      ins(%lhs, %rhs : tensor<2x2x!ttcore.tile<32x32, bf16>>,
                        tensor<2x2x!ttcore.tile<32x32, bf16>>)
      outs(%out_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%lhs_tile: !ttcore.tile<32x32, bf16>,
       %rhs_tile: !ttcore.tile<32x32, bf16>,
       %out_tile: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %sum = ttl.tile_add %lhs_tile, %rhs_tile : !ttcore.tile<32x32, bf16>
    ttl.tile_store %sum, %out_view[%i, %j] : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb1 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// =============================================================================
// Test 2: FPU binary sub (1x1 bf16)
// =============================================================================
// Verifies tile_sub maps to sub_tiles (not sub_binary_tile) on the FPU path.

// FPU-LABEL: func.func @fpu_sub_1x1
// FPU-DAG: %[[C1I:.*]] = arith.constant 1 : i32
// FPU-DAG: %[[C0:.*]] = arith.constant 0 : index
// FPU:     %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// FPU:     %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// FPU:     %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// FPU:     ttkernel.cb_wait_front(%[[CB0]], %[[C1I]])
// FPU:     ttkernel.cb_wait_front(%[[CB2]], %[[C1I]])
// FPU:     ttkernel.cb_reserve_back(%[[CB1]], %[[C1I]])
// FPU:     ttkernel.binary_op_init_common(%[[CB0]], %[[CB2]], %[[CB1]])
// FPU:     ttkernel.tile_regs_acquire
// FPU-NOT: ttkernel.copy_tile
// FPU:     ttkernel.sub_tiles_init(%[[CB0]], %[[CB2]])
// FPU:     ttkernel.sub_tiles(%[[CB0]], %[[CB2]], %[[C0]], %[[C0]], %[[C0]])
// FPU:     ttkernel.tile_regs_commit
// FPU:     ttkernel.tile_regs_wait
// FPU:     ttkernel.pack_tile(%[[C0]], %[[CB1]], %[[C0]], true)
// FPU:     ttkernel.tile_regs_release
// FPU:     ttkernel.cb_push_back(%[[CB1]], %[[C1I]])

// SFPU-LABEL: func.func @fpu_sub_1x1
// SFPU-DAG: %[[C1I:.*]] = arith.constant 1 : i32
// SFPU-DAG: %[[C1:.*]] = arith.constant 1 : index
// SFPU-DAG: %[[C0:.*]] = arith.constant 0 : index
// SFPU:     %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// SFPU:     %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// SFPU:     %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// SFPU:     ttkernel.init_sfpu(%[[CB0]], %[[CB1]])
// SFPU:     ttkernel.tile_regs_acquire
// SFPU:     ttkernel.copy_tile_init(%[[CB0]])
// SFPU:     ttkernel.copy_tile(%[[CB0]], %[[C0]], %[[C0]])
// SFPU:     ttkernel.copy_tile_init(%[[CB2]])
// SFPU:     ttkernel.copy_tile(%[[CB2]], %[[C0]], %[[C1]])
// SFPU:     ttkernel.sub_binary_tile_init
// SFPU:     ttkernel.sub_binary_tile(%[[C0]], %[[C1]], %[[C0]])
// SFPU:     ttkernel.tile_regs_commit
// SFPU:     ttkernel.tile_regs_wait
// SFPU:     ttkernel.pack_tile(%[[C0]], %[[CB1]], %[[C0]], true)
// SFPU:     ttkernel.tile_regs_release
// SFPU:     ttkernel.cb_push_back(%[[CB1]], %[[C1I]])
// SFPU-NOT: ttkernel.sub_tiles

#map_sub = affine_map<(d0, d1) -> (d0, d1)>
func.func @fpu_sub_1x1()
    attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %lhs_ready = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %lhs = ttl.attach_cb %lhs_ready, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rhs_ready = ttl.cb_wait %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rhs = ttl.attach_cb %rhs_ready, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_view = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out = ttl.attach_cb %out_view, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_cb = ttl.attach_cb %empty, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute
      ins(%lhs, %rhs : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                        tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%out_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map_sub, #map_sub, #map_sub],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%lhs_tile: !ttcore.tile<32x32, bf16>,
       %rhs_tile: !ttcore.tile<32x32, bf16>,
       %out_tile: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %diff = ttl.tile_sub %lhs_tile, %rhs_tile : !ttcore.tile<32x32, bf16>
    ttl.tile_store %diff, %out_view[%i, %j] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// =============================================================================
// Test 3: FPU binary mul (1x1 bf16)
// =============================================================================
// Verifies tile_mul maps to mul_tiles (not mul_binary_tile) on the FPU path.

// FPU-LABEL: func.func @fpu_mul_1x1
// FPU-DAG: %[[C1I:.*]] = arith.constant 1 : i32
// FPU-DAG: %[[C0:.*]] = arith.constant 0 : index
// FPU:     %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// FPU:     %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// FPU:     %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// FPU:     ttkernel.cb_wait_front(%[[CB0]], %[[C1I]])
// FPU:     ttkernel.cb_wait_front(%[[CB2]], %[[C1I]])
// FPU:     ttkernel.cb_reserve_back(%[[CB1]], %[[C1I]])
// FPU:     ttkernel.binary_op_init_common(%[[CB0]], %[[CB2]], %[[CB1]])
// FPU:     ttkernel.tile_regs_acquire
// FPU-NOT: ttkernel.copy_tile
// FPU:     ttkernel.mul_tiles_init(%[[CB0]], %[[CB2]])
// FPU:     ttkernel.mul_tiles(%[[CB0]], %[[CB2]], %[[C0]], %[[C0]], %[[C0]])
// FPU:     ttkernel.tile_regs_commit
// FPU:     ttkernel.tile_regs_wait
// FPU:     ttkernel.pack_tile(%[[C0]], %[[CB1]], %[[C0]], true)
// FPU:     ttkernel.tile_regs_release
// FPU:     ttkernel.cb_push_back(%[[CB1]], %[[C1I]])

// SFPU-LABEL: func.func @fpu_mul_1x1
// SFPU-DAG: %[[C1I:.*]] = arith.constant 1 : i32
// SFPU-DAG: %[[C1:.*]] = arith.constant 1 : index
// SFPU-DAG: %[[C0:.*]] = arith.constant 0 : index
// SFPU:     %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// SFPU:     %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// SFPU:     %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// SFPU:     ttkernel.init_sfpu(%[[CB0]], %[[CB1]])
// SFPU:     ttkernel.tile_regs_acquire
// SFPU:     ttkernel.copy_tile_init(%[[CB0]])
// SFPU:     ttkernel.copy_tile(%[[CB0]], %[[C0]], %[[C0]])
// SFPU:     ttkernel.copy_tile_init(%[[CB2]])
// SFPU:     ttkernel.copy_tile(%[[CB2]], %[[C0]], %[[C1]])
// SFPU:     ttkernel.mul_binary_tile_init
// SFPU:     ttkernel.mul_binary_tile(%[[C0]], %[[C1]], %[[C0]])
// SFPU:     ttkernel.tile_regs_commit
// SFPU:     ttkernel.tile_regs_wait
// SFPU:     ttkernel.pack_tile(%[[C0]], %[[CB1]], %[[C0]], true)
// SFPU:     ttkernel.tile_regs_release
// SFPU:     ttkernel.cb_push_back(%[[CB1]], %[[C1I]])
// SFPU-NOT: ttkernel.mul_tiles

#map_mul = affine_map<(d0, d1) -> (d0, d1)>
func.func @fpu_mul_1x1()
    attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %lhs_ready = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %lhs = ttl.attach_cb %lhs_ready, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rhs_ready = ttl.cb_wait %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rhs = ttl.attach_cb %rhs_ready, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_view = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out = ttl.attach_cb %out_view, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_cb = ttl.attach_cb %empty, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute
      ins(%lhs, %rhs : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                        tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%out_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map_mul, #map_mul, #map_mul],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%lhs_tile: !ttcore.tile<32x32, bf16>,
       %rhs_tile: !ttcore.tile<32x32, bf16>,
       %out_tile: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %prod = ttl.tile_mul %lhs_tile, %rhs_tile : !ttcore.tile<32x32, bf16>
    ttl.tile_store %prod, %out_view[%i, %j] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// =============================================================================
// Test 4: FPU binary add + unary tanh chain (2x3 f32)
// =============================================================================
// FPU path: dstPerIteration=1 (FPU add uses 0 DST inputs, tanh in-place).
// f32 capacity=4, 2x3=6 tiles total. Subblock processes 3 tiles per
// iteration with an outer loop of 2 iterations over rows.
//
// SFPU path: dstPerIteration=2 (copy lhs + copy rhs for SFPU add).
// f32 capacity=4, unroll_factor=min(2,6)=2.

// FPU-LABEL: func.func @fpu_add_tanh_f32
// FPU-DAG: %[[C6I:.*]] = arith.constant 6 : i32
// FPU-DAG: %[[C3I:.*]] = arith.constant 3 : i32
// FPU-DAG: %[[C0:.*]] = arith.constant 0 : index
// FPU-DAG: %[[C1:.*]] = arith.constant 1 : index
// FPU-DAG: %[[C2:.*]] = arith.constant 2 : index
// FPU:     %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// FPU:     %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// FPU:     %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// FPU:     ttkernel.cb_wait_front(%[[CB0]], %[[C6I]])
// FPU:     ttkernel.cb_wait_front(%[[CB2]], %[[C6I]])
// FPU:     ttkernel.cb_reserve_back(%[[CB1]], %[[C6I]])
// FPU:     ttkernel.binary_op_init_common(%[[CB0]], %[[CB2]], %[[CB1]])
// Outer loop: 2 iterations (one per row of the 2x3 grid).
// Per-subblock cb_reserve inside loop (outermost dim subblocked).
// FPU:     scf.for %[[IV:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
// FPU:       ttkernel.cb_reserve_back(%[[CB1]], %[[C3I]])
// FPU:       ttkernel.tile_regs_acquire
// Grouped within subblock: all add_tiles, then all tanh_tiles.
// FPU:       %[[IDX0:.*]] = affine.linearize_index [%[[IV]], %[[C0]]] by (2, 3)
// FPU:       ttkernel.add_tiles_init(%[[CB0]], %[[CB2]])
// FPU:       ttkernel.add_tiles(%[[CB0]], %[[CB2]], %[[IDX0]], %[[IDX0]], %[[C0]])
// FPU:       %[[IDX1:.*]] = affine.linearize_index [%[[IV]], %[[C1]]] by (2, 3)
// FPU:       ttkernel.add_tiles(%[[CB0]], %[[CB2]], %[[IDX1]], %[[IDX1]], %[[C1]])
// FPU:       %[[IDX2:.*]] = affine.linearize_index [%[[IV]], %[[C2]]] by (2, 3)
// FPU:       ttkernel.add_tiles(%[[CB0]], %[[CB2]], %[[IDX2]], %[[IDX2]], %[[C2]])
// FPU:       ttkernel.tanh_tile_init
// FPU:       ttkernel.tanh_tile(%[[C0]])
// FPU:       ttkernel.tanh_tile(%[[C1]])
// FPU:       ttkernel.tanh_tile(%[[C2]])
// FPU:       ttkernel.tile_regs_commit
// FPU:       ttkernel.tile_regs_wait
// Pack phase: pack_tile after wait, using local subblock indices.
// FPU:       ttkernel.pack_tile(%[[C0]], %[[CB1]], %[[C0]], true)
// FPU:       ttkernel.pack_tile(%[[C1]], %[[CB1]], %[[C1]], true)
// FPU:       ttkernel.pack_tile(%[[C2]], %[[CB1]], %[[C2]], true)
// FPU:       ttkernel.tile_regs_release
// Per-subblock cb_push inside loop.
// FPU:       ttkernel.cb_push_back(%[[CB1]], %[[C3I]])
// FPU:     } {ttl.subblock_dim = 0 : index, ttl.subblock_loop_stride = 3 : index}
// FPU-NOT: ttkernel.copy_tile
// FPU-NOT: ttkernel.add_binary_tile

// SFPU-LABEL: func.func @fpu_add_tanh_f32
// SFPU-DAG: %[[C6I:.*]] = arith.constant 6 : i32
// SFPU-DAG: %[[C0:.*]] = arith.constant 0 : index
// SFPU-DAG: %[[C1:.*]] = arith.constant 1 : index
// SFPU-DAG: %[[C2:.*]] = arith.constant 2 : index
// SFPU-DAG: %[[C3:.*]] = arith.constant 3 : index
// SFPU:     %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// SFPU:     %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// SFPU:     %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// SFPU:     ttkernel.cb_wait_front(%[[CB0]], %[[C6I]])
// SFPU:     ttkernel.cb_wait_front(%[[CB2]], %[[C6I]])
// SFPU:     ttkernel.cb_reserve_back(%[[CB1]], %[[C6I]])
// SFPU:     ttkernel.init_sfpu(%[[CB0]], %[[CB1]])
// SFPU:     scf.for %[[IV:.*]] = %[[C0]] to %[[C3]] step %[[C1]]
// SFPU:       ttkernel.tile_regs_acquire
// Grouped within subblock: copies from CB0 for both tiles, copies from CB2,
// then adds, then tanhs
// SFPU:       ttkernel.copy_tile_init(%[[CB0]])
// SFPU:       ttkernel.copy_tile(%[[CB0]], %[[IV]], %[[C0]])
// SFPU:       ttkernel.copy_tile(%[[CB0]], {{.*}}, %[[C2]])
// SFPU:       ttkernel.copy_tile_init(%[[CB2]])
// SFPU:       ttkernel.copy_tile(%[[CB2]], %[[IV]], %[[C1]])
// SFPU:       ttkernel.copy_tile(%[[CB2]], {{.*}}, %[[C3]])
// SFPU:       ttkernel.add_binary_tile_init
// SFPU:       ttkernel.add_binary_tile(%[[C0]], %[[C1]], %[[C0]])
// SFPU:       ttkernel.add_binary_tile(%[[C2]], %[[C3]], %[[C2]])
// SFPU:       ttkernel.tanh_tile_init
// SFPU:       ttkernel.tanh_tile(%[[C0]])
// SFPU:       ttkernel.tanh_tile(%[[C2]])
// Pack phase
// SFPU:       ttkernel.tile_regs_commit
// SFPU:       ttkernel.tile_regs_wait
// SFPU:       ttkernel.pack_tile(%[[C0]], %[[CB1]], %[[IV]], true)
// SFPU:       ttkernel.pack_tile(%[[C2]], %[[CB1]], {{.*}}, true)
// SFPU:       ttkernel.tile_regs_release
// SFPU:     } {ttl.subblock_dim = 1 : index, ttl.subblock_loop_stride = 1 : index}
// SFPU:     ttkernel.cb_push_back(%[[CB1]], %[[C6I]])
// SFPU-NOT: ttkernel.add_tiles

#map2 = affine_map<(d0, d1) -> (d0, d1)>
func.func @fpu_add_tanh_f32()
    attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %lhs_ready = ttl.cb_wait %cb0 : <[2, 3], !ttcore.tile<32x32, f32>, 2> -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %lhs = ttl.attach_cb %lhs_ready, %cb0 : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %rhs_ready = ttl.cb_wait %cb2 : <[2, 3], !ttcore.tile<32x32, f32>, 2> -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %rhs = ttl.attach_cb %rhs_ready, %cb2 : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %out_view = ttl.cb_reserve %cb1 : <[2, 3], !ttcore.tile<32x32, f32>, 2> -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %out = ttl.attach_cb %out_view, %cb1 : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %empty = tensor.empty() : tensor<2x3x!ttcore.tile<32x32, f32>>
  %out_cb = ttl.attach_cb %empty, %cb1 : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%lhs, %rhs : tensor<2x3x!ttcore.tile<32x32, f32>>,
                        tensor<2x3x!ttcore.tile<32x32, f32>>)
      outs(%out_cb : tensor<2x3x!ttcore.tile<32x32, f32>>)
      {fp32_dest_acc_en = true,
       indexing_maps = [#map2, #map2, #map2],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%lhs_tile: !ttcore.tile<32x32, f32>,
       %rhs_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %sum = ttl.tile_add %lhs_tile, %rhs_tile : !ttcore.tile<32x32, f32>
    %tanh = ttl.tile_tanh %sum : !ttcore.tile<32x32, f32>
    ttl.tile_store %tanh, %out_view[%i, %j] : !ttcore.tile<32x32, f32>, tensor<2x3x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x3x!ttcore.tile<32x32, f32>>
  ttl.cb_push %cb1 : <[2, 3], !ttcore.tile<32x32, f32>, 2>
  return
}
