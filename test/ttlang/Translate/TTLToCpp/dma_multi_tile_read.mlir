// RUN: ttlang-opt --ttl-to-ttkernel-pipeline --canonicalize %s -o %t.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Test: Multi-tile DMA read (2x2 tiles) tensor → CB.
// Validates nested loop emission through TTL→TTKernel→EmitC→C++.
// Tensor: 64x64xf32 (2x2 tiles), CB: [1,1] (single tile)
// Generated tile loops: for tile_y in 0..2, for tile_x in 0..2
// Tile offset = tile_y * tiles_x + tile_x (row-major ordering)

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: // dma_multi_tile_read
// CHECK: void kernel_main() {
// CHECK-DAG:   size_t [[TILE_STEP:v[0-9]+]] = 1;
// CHECK-DAG:   size_t [[TILES_BOUND:v[0-9]+]] = 2;
// CHECK-DAG:   size_t [[PAGE_SIZE:v[0-9]+]] = 4096;
// CHECK-DAG:   int32_t [[ADDR:v[0-9]+]] = 4096;
// CHECK-DAG:   size_t [[TILE_LB:v[0-9]+]] = 0;
// CHECK:   int32_t [[RT_ARG:v[0-9]+]] = get_common_arg_val<uint32_t>([[TILE_LB]]);
// CHECK:   auto [[ARGS:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<1, 0>();
// CHECK:   TensorAccessor [[ACCESSOR:v[0-9]+]] = TensorAccessor([[ARGS]], [[RT_ARG]], [[ADDR]]);
// CHECK:   int32_t [[CB_PTR:v[0-9]+]] = get_write_ptr(get_compile_time_arg_val(0));
// Cast CB ptr to size_t for index arithmetic
// CHECK:   ptrdiff_t [[CB_PTR_PTRDIFF:v[0-9]+]] = (ptrdiff_t) [[CB_PTR]];
// CHECK:   size_t [[CB_PTR_IDX:v[0-9]+]] = (size_t) [[CB_PTR_PTRDIFF]];
// CHECK:   for (size_t [[TILE_Y:[a-z][0-9]+]] = [[TILE_LB]]; [[TILE_Y]] < [[TILES_BOUND]]; [[TILE_Y]] += [[TILE_STEP]]) {
// CHECK:     for (size_t [[TILE_X:[a-z][0-9]+]] = [[TILE_LB]]; [[TILE_X]] < [[TILES_BOUND]]; [[TILE_X]] += [[TILE_STEP]]) {
// CHECK:       size_t [[TILE_OFFSET_Y:v[0-9]+]] = [[TILE_Y]] * [[TILES_BOUND]];
// CHECK:       size_t [[TILE_OFFSET_X:v[0-9]+]] = [[TILE_OFFSET_Y]] + [[TILE_X]];
// CB address computation: cb_ptr + tile_offset * page_size (all size_t arithmetic)
// CHECK:       size_t [[BYTE_OFF:v[0-9]+]] = [[TILE_OFFSET_X]] * [[PAGE_SIZE]];
// CHECK:       size_t [[CB_ADDR_IDX:v[0-9]+]] = [[CB_PTR_IDX]] + [[BYTE_OFF]];
// Cast to i32 for NOC operation
// CHECK:       ptrdiff_t [[TILE_OFFSET_PTR:v[0-9]+]] = (ptrdiff_t) [[TILE_OFFSET_X]];
// CHECK:       int32_t [[TILE_OFFSET:v[0-9]+]] = (int32_t) [[TILE_OFFSET_PTR]];
// CHECK:       ptrdiff_t [[CB_ADDR_PTR:v[0-9]+]] = (ptrdiff_t) [[CB_ADDR_IDX]];
// CHECK:       int32_t [[CB_ADDR:v[0-9]+]] = (int32_t) [[CB_ADDR_PTR]];
// CHECK:       noc_async_read_tile([[TILE_OFFSET]], [[ACCESSOR]], [[CB_ADDR]]);
// CHECK:     }
// CHECK:   }
// CHECK:   noc_async_read_barrier();
// CHECK:   return;
// CHECK-NEXT: }
module {
  func.func @dma_multi_tile_read(%arg0: tensor<64x64xf32, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>
    %slice = ttl.tensor_slice %arg0[%c0, %c0] : tensor<64x64xf32, #layout> -> !ttl.tensor_slice<tensor<64x64xf32, #layout>>
    %xf = ttl.copy %slice, %cb : (!ttl.tensor_slice<tensor<64x64xf32, #layout>>, !ttl.cb<[2, 2], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}
