// RUN: ttlang-opt --ttl-to-ttkernel-pipeline --canonicalize %s -o %t.ttkernel.mlir
// RUN: ttmlir-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttmlir-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Test: Multiple copies of tensors with the SAME layout but different CB shapes.
// Validates that the same tensor tile grid (2x2) can be copied to different CB configurations.
//
// Both tensors: 64x64xf32 (2x2 tiles) - SAME layout
// Tensor1 → CB1: [2,2] (2x2 tiles, matches tensor grid)
// Tensor2 → CB2: [4,1] (4x1 tiles, different shape from tensor grid)
//
// Expected: Two separate tile loop nests, both iterating 2x2, but with different CB destinations.
// This demonstrates that tile loop bounds are determined by tensor layout, not CB shape.

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: // same_layout_different_cb
// CHECK-NEXT: #include <cstdint>
// CHECK-NEXT: #include "tools/profiler/kernel_profiler.hpp"
// CHECK-NEXT: #include "dataflow_api.h"
// CHECK-NEXT: void kernel_main() {
// CHECK-DAG:   size_t [[TILE_STEP:v[0-9]+]] = 1;
// CHECK-DAG:   size_t [[TILES_BOUND:v[0-9]+]] = 2;
// CHECK-DAG:   int32_t [[ADDR:v[0-9]+]] = 256;
// CHECK-DAG:   size_t [[TILE_LB:v[0-9]+]] = 0;

// First copy: 64x64 (2x2 tiles) → CB [2,2]
// CHECK:   int32_t [[RT_ARG1:v[0-9]+]] = get_common_arg_val<uint32_t>([[TILE_LB]]);
// Placeholder value 42 is a temporary hack, see issue #168
// CHECK:   TensorAccessorArgs [[ACC1_ARGS:v[0-9]+]] = TensorAccessorArgs<42, 0>();
// CHECK:   TensorAccessor [[ACC1:v[0-9]+]] = TensorAccessor([[ACC1_ARGS]], [[RT_ARG1]], [[ADDR]]);
// CHECK:   int32_t [[CB_PTR1:v[0-9]+]] = get_write_ptr(get_compile_time_arg_val(0));
// Generated tile loops iterate over tensor grid (2x2)
// CHECK:   for (size_t [[TILE1_Y:[a-z][0-9]+]] = [[TILE_LB]]; [[TILE1_Y]] < [[TILES_BOUND]]; [[TILE1_Y]] += [[TILE_STEP]]) {
// CHECK:     for (size_t [[TILE1_X:[a-z][0-9]+]] = [[TILE_LB]]; [[TILE1_X]] < [[TILES_BOUND]]; [[TILE1_X]] += [[TILE_STEP]]) {
// CHECK:       size_t [[TILE1_OFFSET_Y:v[0-9]+]] = [[TILE1_Y]] * [[TILES_BOUND]];
// CHECK:       size_t [[TILE1_OFFSET_X:v[0-9]+]] = [[TILE1_OFFSET_Y]] + [[TILE1_X]];
// CHECK:       ptrdiff_t [[TILE1_OFFSET_PTR:v[0-9]+]] = (ptrdiff_t) [[TILE1_OFFSET_X]];
// CHECK:       int32_t [[TILE1_OFFSET:v[0-9]+]] = (int32_t) [[TILE1_OFFSET_PTR]];
// CHECK:       noc_async_read_tile([[TILE1_OFFSET]], [[ACC1]], [[CB_PTR1]]);
// CHECK:     }
// CHECK:   }
// CHECK:   noc_async_read_barrier();

// Second copy: 64x64 (2x2 tiles) → CB [4,1] - SAME tensor layout, DIFFERENT CB shape
// CHECK:   int32_t [[RT_ARG2:v[0-9]+]] = get_common_arg_val<uint32_t>([[TILE_STEP]]);
// Placeholder value 43 is a temporary hack, see issue #168
// CHECK:   TensorAccessorArgs [[ACC2_ARGS:v[0-9]+]] = TensorAccessorArgs<43, 0>();
// CHECK:   TensorAccessor [[ACC2:v[0-9]+]] = TensorAccessor([[ACC2_ARGS]], [[RT_ARG2]], [[ADDR]]);
// CHECK:   int32_t [[CB_PTR2:v[0-9]+]] = get_write_ptr(get_compile_time_arg_val(1));
// Generated tile loops still iterate over tensor grid (2x2), not CB shape (4x1)
// CHECK:   for (size_t [[TILE2_Y:[a-z][0-9]+]] = [[TILE_LB]]; [[TILE2_Y]] < [[TILES_BOUND]]; [[TILE2_Y]] += [[TILE_STEP]]) {
// CHECK:     for (size_t [[TILE2_X:[a-z][0-9]+]] = [[TILE_LB]]; [[TILE2_X]] < [[TILES_BOUND]]; [[TILE2_X]] += [[TILE_STEP]]) {
// CHECK:       size_t [[TILE2_OFFSET_Y:v[0-9]+]] = [[TILE2_Y]] * [[TILES_BOUND]];
// CHECK:       size_t [[TILE2_OFFSET_X:v[0-9]+]] = [[TILE2_OFFSET_Y]] + [[TILE2_X]];
// CHECK:       ptrdiff_t [[TILE2_OFFSET_PTR:v[0-9]+]] = (ptrdiff_t) [[TILE2_OFFSET_X]];
// CHECK:       int32_t [[TILE2_OFFSET:v[0-9]+]] = (int32_t) [[TILE2_OFFSET_PTR]];
// CHECK:       noc_async_read_tile([[TILE2_OFFSET]], [[ACC2]], [[CB_PTR2]]);
// CHECK:     }
// CHECK:   }
// CHECK:   noc_async_read_barrier();
// CHECK:   return;
// CHECK-NEXT: }

module {
  func.func @same_layout_different_cb(%arg0: tensor<64x64xf32, #layout>, %arg1: tensor<64x64xf32, #layout>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb1 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>
    %cb2 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4, 1], f32, 2>

    // First copy: 64x64 → CB [2,2]
    %xf1 = ttl.copy %arg0, %cb1 : (tensor<64x64xf32, #layout>, !ttl.cb<[2, 2], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf1 : !ttl.transfer_handle<read>

    // Second copy: 64x64 (same layout) → CB [4,1] (different CB shape)
    %xf2 = ttl.copy %arg1, %cb2 : (tensor<64x64xf32, #layout>, !ttl.cb<[4, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf2 : !ttl.transfer_handle<read>

    func.return
  }
}
