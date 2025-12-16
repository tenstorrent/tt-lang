// RUN: ttlang-opt --ttl-to-ttkernel-pipeline="fuse-tile-loops=true" --canonicalize %s -o %t.ttkernel.mlir
// RUN: ttmlir-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttmlir-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Test: 1-D tile grid fusion (single row of tiles).
// Tensors: 32x64xf32 = 1x2 tiles (1 row, 2 columns)
// Both copies fused into a single tile loop.
//
// Lowering generates 2D loops (y=0..1, x=0..2), but canonicalization removes
// the trivial y loop (iterates only once), leaving just: for tile_x in 0..2
//
// After fusion:
//   for tile_x in 0..2:
//     noc_async_read_tile(tile_x, accessor1, ...)
//     noc_async_read_tile(tile_x, accessor2, ...)
//   noc_async_read_barrier()
//   noc_async_read_barrier()

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: // multi_tile_1d_fused
// CHECK-NEXT: #include <cstdint>
// CHECK-NEXT: #include "tools/profiler/kernel_profiler.hpp"
// CHECK-NEXT: #include "dataflow_api.h"
// CHECK-NEXT: void kernel_main() {
// CHECK-DAG:   size_t [[TILES_X:v[0-9]+]] = 2;
// CHECK-DAG:   size_t [[STEP:v[0-9]+]] = 1;
// CHECK-DAG:   size_t [[LB:v[0-9]+]] = 0;
// CHECK-DAG:   int32_t [[ZERO:v[0-9]+]] = 0;

// Fused tile loop: single loop for 1x2 grid with both DMAs in body
// (y-loop removed by canonicalization since it only iterates once)
// CHECK:   for (size_t [[TILE_X:[a-z][0-9]+]] = [[LB]]; [[TILE_X]] < [[TILES_X]]; [[TILE_X]] += [[STEP]]) {
// CHECK:       noc_async_read_tile({{.*}}, {{.*}}, [[ZERO]]);
// CHECK-NEXT:  noc_async_read_tile({{.*}}, {{.*}}, [[ZERO]]);
// CHECK:     }

// Consecutive barriers deduplicated to single barrier.
// CHECK:   noc_async_read_barrier();
// CHECK:   return;
// CHECK-NEXT: }

module {
  func.func @multi_tile_1d_fused(%arg0: tensor<32x64xf32, #layout>, %arg1: tensor<32x64xf32, #layout>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb1 = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb2 = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>

    // Both tensors have same 1D tile grid (1x2) - will be fused into single loop
    %xf1 = ttl.copy %arg0, %cb1 : (tensor<32x64xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf2 = ttl.copy %arg1, %cb2 : (tensor<32x64xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>

    // Barriers wait for both transfers
    ttl.wait %xf1 : !ttl.transfer_handle<read>
    ttl.wait %xf2 : !ttl.transfer_handle<read>

    func.return
  }
}
