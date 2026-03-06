// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-subblock-compute-for-dst, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-schedule-operations, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   | FileCheck %s
// Purpose: Verify f32 2x3 subblocking picks a DST-safe subblock size.
//
// With f32 tiles (fp32_dest_acc_en=true), DST capacity is 4 half-tiles.
// The add+tanh compute uses 2 DST slots per tile (copy lhs + copy rhs).
// TTLAssignDST computes unrollFactor = capacity / dstPerIteration = 4 / 2 = 2.
// TTLSubblockComputeForDST picks a [2, 1] subblock (2 tiles), which fits
// within the 4-slot f32 capacity (2 tiles * 2 copies = 4 DST slots).
//
// After scheduling, copy_tile ops from cb0 are grouped, then copy_tile ops
// from cb1, then adds, then tanhs. With a 2-tile subblock this uses 4 DST
// slots (exactly at capacity).
//
// The scheduling pass cannot cause DST overflow when subblocking is correct:
// peak_after_scheduling = N * dstPerIteration <= capacity, because
// N = floor(capacity / dstPerIteration) by construction.
// CHECK-LABEL: func.func @f32_subblock_scheduling
// CHECK: ttkernel.tile_regs_acquire
// After scheduling: copies grouped by CB, then adds, then tanhs.
// 2 tiles * 2 copies = 4 copy_tile ops per sync region (within f32 capacity).
// CHECK:       ttkernel.copy_tile_init(
// CHECK-NEXT:  ttkernel.copy_tile(
// CHECK-NEXT:  ttkernel.copy_tile(
// CHECK-NEXT:  ttkernel.copy_tile_init(
// CHECK-NEXT:  ttkernel.copy_tile(
// CHECK-NEXT:  ttkernel.copy_tile(
// CHECK-NEXT:  ttkernel.add_binary_tile_init
// CHECK-NEXT:  ttkernel.add_binary_tile(
// CHECK-NEXT:  ttkernel.add_binary_tile(
// CHECK-NEXT:  ttkernel.tanh_tile_init
// CHECK-NEXT:  ttkernel.tanh_tile(
// CHECK-NEXT:  ttkernel.tanh_tile(
// CHECK-NEXT:  ttkernel.tile_regs_commit
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @f32_subblock_scheduling()
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
