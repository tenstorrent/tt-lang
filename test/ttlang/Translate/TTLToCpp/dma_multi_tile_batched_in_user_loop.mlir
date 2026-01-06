// RUN: ttlang-opt --ttl-to-ttkernel-pipeline --canonicalize %s -o %t.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Test: Multi-tile copies (same tile grid) within a user loop.
// Multiple copies with SAME tile grid dimensions within a user loop.
//
// User loop: 0..3
// Both tensors: 64x64xf32 (2x2 tiles) - SAME tile grid
// Both copies issued before barriers
//
// Current behavior (generates separate tile loops):
//   for user_iter in 0..3:
//     for tile_y in 0..2:
//       for tile_x in 0..2:
//         noc_async_read_tile(offset, accessor1, ...)
//     for tile_y in 0..2:
//       for tile_x in 0..2:
//         noc_async_read_tile(offset, accessor2, ...)
//     noc_async_read_barrier()
//     noc_async_read_barrier()
//
// Future optimization: Custom loop fusion pass could merge loops with identical bounds
// to batch both DMAs in the same tile loop body. MLIR's built-in --affine-loop-fusion
// only works for affine dialect loops, and --scf-parallel-loop-fusion only works for
// scf.parallel loops. We must implement custom fusion for this.

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: // batched_multi_tile_user_loop
// CHECK: void kernel_main() {
// CHECK-DAG:   size_t [[TILES_BOUND:v[0-9]+]] = 2;
// CHECK-DAG:   int32_t [[ADDR:v[0-9]+]] = 4096;
// CHECK-DAG:   size_t [[STEP:v[0-9]+]] = 1;
// CHECK-DAG:   size_t [[USER_UB:v[0-9]+]] = 3;
// CHECK-DAG:   size_t [[LB:v[0-9]+]] = 0;

// Accessors materialized at function entry with chaining
// CHECK:   int32_t [[RT_ARG1:v[0-9]+]] = get_common_arg_val<uint32_t>([[LB]]);
// First accessor uses literal base CTA index = num_cbs = 2
// CHECK:   auto [[ACC1_ARGS:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<2, 0>();
// CHECK:   TensorAccessor [[ACC1:v[0-9]+]] = TensorAccessor([[ACC1_ARGS]], [[RT_ARG1]], [[ADDR]]);
// Second accessor chains from first
// CHECK:   int32_t [[RT_ARG2:v[0-9]+]] = get_common_arg_val<uint32_t>([[STEP]]);
// CHECK:   auto [[ACC2_ARGS:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<[[ACC1_ARGS]].next_compile_time_args_offset(), [[ACC1_ARGS]].next_common_runtime_args_offset()>();
// CHECK:   TensorAccessor [[ACC2:v[0-9]+]] = TensorAccessor([[ACC2_ARGS]], [[RT_ARG2]], [[ADDR]]);

// User loop from input MLIR (0..3)
// CHECK:   for (size_t [[USER_ITER:[a-z][0-9]+]] = [[LB]]; [[USER_ITER]] < [[USER_UB]]; [[USER_ITER]] += [[STEP]]) {

// First copy: 64x64 (2x2 tiles) → CB1, uses pre-materialized accessor ACC1
// CHECK-NOT:     TensorAccessorArgs
// CHECK-NOT:     TensorAccessor
// CHECK:     int32_t [[CB_PTR1:v[0-9]+]] = get_write_ptr(get_compile_time_arg_val(0));
// Tile loops: for tile_y in 0..2, for tile_x in 0..2
// CHECK:     for (size_t [[TILE1_Y:[a-z][0-9]+]] = [[LB]]; [[TILE1_Y]] < [[TILES_BOUND]]; [[TILE1_Y]] += [[STEP]]) {
// CHECK:       for (size_t [[TILE1_X:[a-z][0-9]+]] = [[LB]]; [[TILE1_X]] < [[TILES_BOUND]]; [[TILE1_X]] += [[STEP]]) {
// CHECK:         size_t [[TILE1_OFFSET_Y:v[0-9]+]] = [[TILE1_Y]] * [[TILES_BOUND]];
// CHECK:         size_t [[TILE1_OFFSET_X:v[0-9]+]] = [[TILE1_OFFSET_Y]] + [[TILE1_X]];
// CHECK:         ptrdiff_t [[TILE1_OFFSET_PTR:v[0-9]+]] = (ptrdiff_t) [[TILE1_OFFSET_X]];
// CHECK:         int32_t [[TILE1_OFFSET:v[0-9]+]] = (int32_t) [[TILE1_OFFSET_PTR]];
// CHECK:         noc_async_read_tile([[TILE1_OFFSET]], [[ACC1]], [[CB_PTR1]]);
// CHECK:       }
// CHECK:     }

// Second copy: 64x64 (2x2 tiles) → CB2, uses pre-materialized accessor ACC2
// CHECK-NOT:     TensorAccessorArgs
// CHECK-NOT:     TensorAccessor
// CHECK:     int32_t [[CB_PTR2:v[0-9]+]] = get_write_ptr(get_compile_time_arg_val(1));
// Separate tile loops (same bounds 0..2 x 0..2 but not merged with first copy)
// CHECK:     for (size_t [[TILE2_Y:[a-z][0-9]+]] = [[LB]]; [[TILE2_Y]] < [[TILES_BOUND]]; [[TILE2_Y]] += [[STEP]]) {
// CHECK:       for (size_t [[TILE2_X:[a-z][0-9]+]] = [[LB]]; [[TILE2_X]] < [[TILES_BOUND]]; [[TILE2_X]] += [[STEP]]) {
// CHECK:         size_t [[TILE2_OFFSET_Y:v[0-9]+]] = [[TILE2_Y]] * [[TILES_BOUND]];
// CHECK:         size_t [[TILE2_OFFSET_X:v[0-9]+]] = [[TILE2_OFFSET_Y]] + [[TILE2_X]];
// CHECK:         ptrdiff_t [[TILE2_OFFSET_PTR:v[0-9]+]] = (ptrdiff_t) [[TILE2_OFFSET_X]];
// CHECK:         int32_t [[TILE2_OFFSET:v[0-9]+]] = (int32_t) [[TILE2_OFFSET_PTR]];
// CHECK:         noc_async_read_tile([[TILE2_OFFSET]], [[ACC2]], [[CB_PTR2]]);
// CHECK:       }
// CHECK:     }

// Consecutive barriers deduplicated to single barrier.
// CHECK:     noc_async_read_barrier();
// CHECK:   }
// CHECK:   return;
// CHECK-NEXT: }

module {
  func.func @batched_multi_tile_user_loop(%arg0: tensor<64x64xf32, #layout>, %arg1: tensor<64x64xf32, #layout>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.base_cta_index = 2 : i32} {
    %c0 = arith.constant 0 : index
    %cb1 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb2 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index

    // User loop (0..3)
    scf.for %i = %c0 to %c3 step %c1 {
      // Batch: issue both copies before barrier
      // Both tensors have same tile grid (2x2) - potential for shared tile loop
      %xf1 = ttl.copy %arg0, %cb1 : (tensor<64x64xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
      %xf2 = ttl.copy %arg1, %cb2 : (tensor<64x64xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>

      // Single barrier waits for both transfers
      ttl.wait %xf1 : !ttl.transfer_handle<read>
      ttl.wait %xf2 : !ttl.transfer_handle<read>
    }

    func.return
  }
}
