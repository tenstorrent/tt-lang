// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-subblock-compute-for-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s
// XFAIL: *
// Purpose: Regression test for f32 2x3 subblocking + scheduling overflow.
//
// With f32 tiles (fp32_dest_acc_en=true), DST capacity is 4 half-tiles.
// The add+tanh compute uses 2 DST slots per tile (copy lhs + copy rhs).
// TTLAssignDST computes unrollFactor = capacity / dstPerIteration = 4 / 2 = 2.
// TTLSubblockComputeForDST should pick a subblock of at most 2 tiles, but
// for the 2x3 geometry it picks 1x3 = 3 tiles, exceeding the budget.
//
// Without scheduling, the interleaved per-tile order (copy, copy, add, tanh)
// reuses DST slots sequentially and happens to work. With scheduling, all
// copy_tile ops are grouped first, requiring all 6 DST slots (3 tiles * 2
// slots) to be live simultaneously, overflowing the 4-slot f32 capacity.
//
// Expected: after scheduling, copy_tile ops from cb0 should be grouped,
// then copy_tile ops from cb1, then adds, then tanhs. With a correct
// subblock of 2 tiles, this would use 4 DST slots (within capacity).
//
// The bug is in TTLSubblockComputeForDST picking 1x3 instead of 1x2.
// This test XFAILs until that is fixed.
// CHECK-LABEL: func.func @f32_subblock_overflow
// CHECK: ttkernel.tile_regs_acquire
// We expect at most 4 copy_tile ops per sync region (2 tiles * 2 copies).
// With the bug, we get 6 (3 tiles * 2 copies), overflowing DST.
// CHECK-COUNT-4: ttkernel.copy_tile
// CHECK-NOT: ttkernel.copy_tile
// CHECK: ttkernel.tile_regs_commit
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @f32_subblock_overflow()
    attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
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
       indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%lhs_tile: !ttcore.tile<32x32, f32>,
       %rhs_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %lhs_tile, %rhs_tile : !ttcore.tile<32x32, f32>
    %tanh = ttl.tile_tanh %sum : !ttcore.tile<32x32, f32>
    ttl.tile_store %tanh, %out_view : !ttcore.tile<32x32, f32>, tensor<2x3x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x3x!ttcore.tile<32x32, f32>>
  ttl.cb_push %cb1 : <[2, 3], !ttcore.tile<32x32, f32>, 2>
  ttl.cb_pop %cb2 : <[2, 3], !ttcore.tile<32x32, f32>, 2>
  ttl.cb_pop %cb0 : <[2, 3], !ttcore.tile<32x32, f32>, 2>
  return
}
