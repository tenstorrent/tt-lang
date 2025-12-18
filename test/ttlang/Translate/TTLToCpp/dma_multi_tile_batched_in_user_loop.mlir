// RUN: ttlang-opt --ttl-to-ttkernel-pipeline --canonicalize %s -o %t.ttkernel.mlir
// RUN: ttmlir-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttmlir-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
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
// CHECK-NEXT: #include <cstdint>
// CHECK-NEXT: #include "tools/profiler/kernel_profiler.hpp"
// CHECK-NEXT: #include "dataflow_api.h"
// CHECK-NEXT: void kernel_main() {
// CHECK:   for (size_t [[USER_ITER:[a-z][0-9]+]] = {{.*}}; [[USER_ITER]] < {{.*}}; [[USER_ITER]] += {{.*}}) {
// CHECK-NEXT:     TensorAccessorArgs [[ACC1_ARGS:v[0-9]+]] = TensorAccessorArgs<64, 1>();
// CHECK-NEXT:     TensorAccessor [[ACC1:v[0-9]+]] = TensorAccessor([[ACC1_ARGS]], {{.*}}, {{.*}});
// CHECK-NEXT:     for (size_t [[T1_Y:[a-z][0-9]+]] = {{.*}}; [[T1_Y]] < {{.*}}; [[T1_Y]] += {{.*}}) {
// CHECK-NEXT:       for (size_t [[T1_X:[a-z][0-9]+]] = {{.*}}; [[T1_X]] < {{.*}}; [[T1_X]] += {{.*}}) {
// CHECK-NEXT:         size_t [[T1_LINEAR:v[0-9]+]] = [[T1_Y]] * {{.*}};
// CHECK-NEXT:         size_t [[T1_OFFSET:v[0-9]+]] = [[T1_LINEAR]] + [[T1_X]];
// CHECK-NEXT:         ptrdiff_t [[T1_PTR:v[0-9]+]] = (ptrdiff_t) [[T1_OFFSET]];
// CHECK-NEXT:         int32_t [[T1_TILE:v[0-9]+]] = (int32_t) [[T1_PTR]];
// CHECK-NEXT:         noc_async_read_tile([[T1_TILE]], [[ACC1]], {{.*}});
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     TensorAccessorArgs [[ACC2_ARGS:v[0-9]+]] = TensorAccessorArgs<64, 1>();
// CHECK-NEXT:     TensorAccessor [[ACC2:v[0-9]+]] = TensorAccessor([[ACC2_ARGS]], {{.*}}, {{.*}});
// CHECK-NEXT:     for (size_t [[T2_Y:[a-z][0-9]+]] = {{.*}}; [[T2_Y]] < {{.*}}; [[T2_Y]] += {{.*}}) {
// CHECK-NEXT:       for (size_t [[T2_X:[a-z][0-9]+]] = {{.*}}; [[T2_X]] < {{.*}}; [[T2_X]] += {{.*}}) {
// CHECK-NEXT:         size_t [[T2_LINEAR:v[0-9]+]] = [[T2_Y]] * {{.*}};
// CHECK-NEXT:         size_t [[T2_OFFSET:v[0-9]+]] = [[T2_LINEAR]] + [[T2_X]];
// CHECK-NEXT:         ptrdiff_t [[T2_PTR:v[0-9]+]] = (ptrdiff_t) [[T2_OFFSET]];
// CHECK-NEXT:         int32_t [[T2_TILE:v[0-9]+]] = (int32_t) [[T2_PTR]];
// CHECK-NEXT:         noc_async_read_tile([[T2_TILE]], [[ACC2]], {{.*}});
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     noc_async_read_barrier();
// CHECK-NEXT:   }
// CHECK-NEXT:   return;
// CHECK-NEXT: }

module {
  func.func @batched_multi_tile_user_loop(%arg0: tensor<64x64xf32, #layout>, %arg1: tensor<64x64xf32, #layout>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
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
