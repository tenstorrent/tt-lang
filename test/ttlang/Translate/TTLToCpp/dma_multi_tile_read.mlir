// RUN: ttlang-opt --ttl-to-ttkernel-pipeline --canonicalize %s -o %t.ttkernel.mlir
// RUN: ttmlir-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttmlir-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Test: Multi-tile DMA read (2x2 tiles) tensor → CB.
// Validates nested loop emission through TTL→TTKernel→EmitC→C++.
// Tensor: 64x64xf32 (2x2 tiles), CB: [1,1] (single tile)
// Generated tile loops: for tile_y in 0..2, for tile_x in 0..2
// Tile offset = tile_y * tiles_x + tile_x (row-major ordering)

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: // dma_multi_tile_read
// CHECK-NEXT: #include <cstdint>
// CHECK-NEXT: #include "tools/profiler/kernel_profiler.hpp"
// CHECK-NEXT: #include "dataflow_api.h"
// CHECK-NEXT: void kernel_main() {
// CHECK-DAG:   size_t [[TILE_STEP:v[0-9]+]] = 1;
// CHECK-DAG:   size_t [[TILES_BOUND:v[0-9]+]] = 2;
// CHECK-DAG:   int32_t [[ADDR:v[0-9]+]] = 256;
// CHECK-DAG:   size_t [[TILE_LB:v[0-9]+]] = 0;
// CHECK:   int32_t [[RT_ARG:v[0-9]+]] = get_common_arg_val<uint32_t>([[TILE_LB]]);
// CHECK:   TensorAccessorArgs [[ARGS:v[0-9]+]] = TensorAccessorArgs<64, 1>();
// CHECK:   TensorAccessor [[ACCESSOR:v[0-9]+]] = TensorAccessor([[ARGS]], [[RT_ARG]], [[ADDR]]);
// CHECK:   int32_t [[CB_PTR:v[0-9]+]] = get_write_ptr(get_compile_time_arg_val(0));
// CHECK:   for (size_t [[TILE_Y:[a-z][0-9]+]] = [[TILE_LB]]; [[TILE_Y]] < [[TILES_BOUND]]; [[TILE_Y]] += [[TILE_STEP]]) {
// CHECK:     for (size_t [[TILE_X:[a-z][0-9]+]] = [[TILE_LB]]; [[TILE_X]] < [[TILES_BOUND]]; [[TILE_X]] += [[TILE_STEP]]) {
// CHECK:       size_t [[TILE_OFFSET_Y:v[0-9]+]] = [[TILE_Y]] * [[TILES_BOUND]];
// CHECK:       size_t [[TILE_OFFSET_X:v[0-9]+]] = [[TILE_OFFSET_Y]] + [[TILE_X]];
// CHECK:       ptrdiff_t [[TILE_OFFSET_PTR:v[0-9]+]] = (ptrdiff_t) [[TILE_OFFSET_X]];
// CHECK:       int32_t [[TILE_OFFSET:v[0-9]+]] = (int32_t) [[TILE_OFFSET_PTR]];
// CHECK:       noc_async_read_tile([[TILE_OFFSET]], [[ACCESSOR]], [[CB_PTR]]);
// CHECK:     }
// CHECK:   }
// CHECK:   noc_async_read_barrier();
// CHECK:   return;
// CHECK-NEXT: }
module {
  func.func @dma_multi_tile_read(%arg0: tensor<64x64xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %arg0, %cb : (tensor<64x64xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}
