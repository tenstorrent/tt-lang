// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-tile-and-assign-dst,ttl-insert-tile-regs-sync,ttl-lower-to-loops{unroll-compute=0},ttl-annotate-cb-associations),convert-ttl-to-ttkernel,canonicalize,cse,lower-affine)' \
// RUN:   -o %t.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Purpose: Complete example with reader, compute, and writer threads.
// Pattern: reader (NOC) → CBs → compute (MATH) → CB → writer (NOC)
// Operation: f(A + B) where f is exp, matching the C++ example pattern.

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: // reader_binary
// CHECK: void kernel_main() {
// CHECK-DAG:   size_t [[ONE:.*]] = 1;
// CHECK-DAG:   size_t [[BOUND:.*]] = 2;
// CHECK-DAG:   size_t [[PAGE_SIZE:.*]] = 4096;
// CHECK-DAG:   size_t [[ZERO:.*]] = 0;

// Read tensor A into CB0
// CHECK:   int32_t [[RT_ARG_A:.*]] = get_common_arg_val<uint32_t>([[ZERO]]);
// CHECK-NEXT:   auto [[ARGS_A:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<2, 0>();
// CHECK-NEXT:   TensorAccessor [[ACC_A:.*]] = TensorAccessor([[ARGS_A]], [[RT_ARG_A]],
// CHECK:   int32_t [[CB0_PTR:.*]] = get_write_ptr(get_compile_time_arg_val(0));
// Cast CB ptr to size_t for index arithmetic
// CHECK-NEXT:   ptrdiff_t [[CB0_PTR_PTRDIFF:v[0-9]+]] = (ptrdiff_t) [[CB0_PTR]];
// CHECK-NEXT:   size_t [[CB0_PTR_IDX:v[0-9]+]] = (size_t) [[CB0_PTR_PTRDIFF]];
// CHECK-NEXT:   for (size_t [[I_A:.*]] = [[ZERO]]; [[I_A]] < [[BOUND]]; [[I_A]] += [[ONE]]) {
// CHECK-NEXT:     for (size_t [[J_A:.*]] = [[ZERO]]; [[J_A]] < [[BOUND]]; [[J_A]] += [[ONE]]) {
// Tile offset computation: i * cols + j
// CHECK:       size_t [[TILE_OFF_A_Y:v[0-9]+]] = [[I_A]] * [[BOUND]];
// CHECK-NEXT:       size_t [[TILE_OFF_A_X:v[0-9]+]] = [[TILE_OFF_A_Y]] + [[J_A]];
// CB address computation: cb_ptr + tile_offset * page_size (all size_t arithmetic)
// CHECK-NEXT:       size_t [[BYTE_OFF_A:v[0-9]+]] = [[TILE_OFF_A_X]] * [[PAGE_SIZE]];
// CHECK-NEXT:       size_t [[CB_ADDR_A_IDX:v[0-9]+]] = [[CB0_PTR_IDX]] + [[BYTE_OFF_A]];
// Cast to i32 for NOC operation
// CHECK-NEXT:       ptrdiff_t [[TILE_OFF_A_PTR:v[0-9]+]] = (ptrdiff_t) [[TILE_OFF_A_X]];
// CHECK-NEXT:       int32_t [[TILE_OFF_A:v[0-9]+]] = (int32_t) [[TILE_OFF_A_PTR]];
// CHECK-NEXT:       ptrdiff_t [[CB_ADDR_A_PTR:v[0-9]+]] = (ptrdiff_t) [[CB_ADDR_A_IDX]];
// CHECK-NEXT:       int32_t [[CB_ADDR_A:v[0-9]+]] = (int32_t) [[CB_ADDR_A_PTR]];
// CHECK-NEXT:       noc_async_read_tile([[TILE_OFF_A]], [[ACC_A]], [[CB_ADDR_A]]);
// CHECK:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   noc_async_read_barrier();

// Read tensor B into CB1
// CHECK:   int32_t [[RT_ARG_B:.*]] = get_common_arg_val<uint32_t>([[ONE]]);
// CHECK-NEXT:   auto [[ARGS_B:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<3, 1>();
// CHECK-NEXT:   TensorAccessor [[ACC_B:.*]] = TensorAccessor([[ARGS_B]], [[RT_ARG_B]],
// CHECK:   int32_t [[CB1_PTR:.*]] = get_write_ptr(get_compile_time_arg_val(1));
// Cast CB ptr to size_t for index arithmetic
// CHECK-NEXT:   ptrdiff_t [[CB1_PTR_PTRDIFF:v[0-9]+]] = (ptrdiff_t) [[CB1_PTR]];
// CHECK-NEXT:   size_t [[CB1_PTR_IDX:v[0-9]+]] = (size_t) [[CB1_PTR_PTRDIFF]];
// CHECK-NEXT:   for (size_t [[I_B:.*]] = [[ZERO]]; [[I_B]] < [[BOUND]]; [[I_B]] += [[ONE]]) {
// CHECK-NEXT:     for (size_t [[J_B:.*]] = [[ZERO]]; [[J_B]] < [[BOUND]]; [[J_B]] += [[ONE]]) {
// Tile offset computation: i * cols + j
// CHECK:       size_t [[TILE_OFF_B_Y:v[0-9]+]] = [[I_B]] * [[BOUND]];
// CHECK-NEXT:       size_t [[TILE_OFF_B_X:v[0-9]+]] = [[TILE_OFF_B_Y]] + [[J_B]];
// CB address computation: cb_ptr + tile_offset * page_size (all size_t arithmetic)
// CHECK-NEXT:       size_t [[BYTE_OFF_B:v[0-9]+]] = [[TILE_OFF_B_X]] * [[PAGE_SIZE]];
// CHECK-NEXT:       size_t [[CB_ADDR_B_IDX:v[0-9]+]] = [[CB1_PTR_IDX]] + [[BYTE_OFF_B]];
// Cast to i32 for NOC operation
// CHECK-NEXT:       ptrdiff_t [[TILE_OFF_B_PTR:v[0-9]+]] = (ptrdiff_t) [[TILE_OFF_B_X]];
// CHECK-NEXT:       int32_t [[TILE_OFF_B:v[0-9]+]] = (int32_t) [[TILE_OFF_B_PTR]];
// CHECK-NEXT:       ptrdiff_t [[CB_ADDR_B_PTR:v[0-9]+]] = (ptrdiff_t) [[CB_ADDR_B_IDX]];
// CHECK-NEXT:       int32_t [[CB_ADDR_B:v[0-9]+]] = (int32_t) [[CB_ADDR_B_PTR]];
// CHECK-NEXT:       noc_async_read_tile([[TILE_OFF_B]], [[ACC_B]], [[CB_ADDR_B]]);
// CHECK:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   noc_async_read_barrier();
// CHECK-NEXT:   return;
// CHECK-NEXT: }

// Reader kernel: reads A and B from DRAM, pushes to CB0 and CB1
func.func @reader_binary(%a: tensor<64x64xf32, #layout>, %b: tensor<64x64xf32, #layout>)
    attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [0, 1], ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>

  // Copy A to CB0
  %slice_a = ttl.tensor_slice %a[%c0, %c0] : tensor<64x64xf32, #layout> -> tensor<64x64xf32, #layout>
  %xf_a = ttl.copy %slice_a, %cb0 : (tensor<64x64xf32, #layout>, !ttl.cb<[2, 2], f32, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %xf_a : !ttl.transfer_handle<read>

  // Copy B to CB1
  %slice_b = ttl.tensor_slice %b[%c0, %c0] : tensor<64x64xf32, #layout> -> tensor<64x64xf32, #layout>
  %xf_b = ttl.copy %slice_b, %cb1 : (tensor<64x64xf32, #layout>, !ttl.cb<[2, 2], f32, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %xf_b : !ttl.transfer_handle<read>

  func.return
}

// CHECK-LABEL: // compute_fused
// CHECK: void kernel_main() {
// CHECK-DAG:   int32_t [[TILES:.*]] = 4;
// CHECK-DAG:   size_t [[BOUND:.*]] = 2;
// CHECK-DAG:   size_t [[ONE:.*]] = 1;
// CHECK-DAG:   size_t [[ZERO:.*]] = 0;

// Wait for inputs from reader
// CHECK:   cb_wait_front(get_compile_time_arg_val(0), [[TILES]]);
// CHECK-NEXT:   cb_wait_front(get_compile_time_arg_val(1), [[TILES]]);

// Initialize SFPU for CB data formats
// CHECK-NEXT:   init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(2));

// Nested loops over 2x2 tile grid
// CHECK-NEXT:   for (size_t [[I:.*]] = [[ZERO]]; [[I]] < [[BOUND]]; [[I]] += [[ONE]]) {
// CHECK-NEXT:     for (size_t [[J:.*]] = [[ZERO]]; [[J]] < [[BOUND]]; [[J]] += [[ONE]]) {

// Compute linear tile index: i * cols + j
// CHECK:            size_t [[COL_SIZE:.*]] = 2;
// CHECK-NEXT:       size_t [[IOFF:.*]] = [[I]] * [[COL_SIZE]];
// CHECK-NEXT:       size_t [[LINIDX:.*]] = [[IOFF]] + [[J]];

// Compute CB tile index: i * 2 + j (linearized row-major index)
// CB index is computed from IVs before tile_regs_acquire for use in pack_tile.
// CHECK-NEXT:       size_t [[CB_OFF_I:v[0-9]+]] = [[I]] * {{.*}};
// CHECK-NEXT:       size_t [[CB_IDX:v[0-9]+]] = [[CB_OFF_I]] + [[J]];

// Acquire DST registers (inside loop)
// CHECK-NEXT:       tile_regs_acquire();

// Load tile from CB0 into DST[0]
// CHECK-NEXT:       copy_tile_init(get_compile_time_arg_val(0));
// CHECK-NEXT:       copy_tile(get_compile_time_arg_val(0), [[LINIDX]], [[ZERO]]);

// Load tile from CB1 into DST[1]
// CHECK-NEXT:       copy_tile_init(get_compile_time_arg_val(1));
// CHECK-NEXT:       copy_tile(get_compile_time_arg_val(1), [[LINIDX]], [[ONE]]);

// Compute: A + B
// CHECK-NEXT:       add_binary_tile_init();
// CHECK-NEXT:       add_binary_tile([[ZERO]], [[ONE]], [[ZERO]]);

// Compute: exp(A + B)
// CHECK-NEXT:       exp_tile_init();
// CHECK-NEXT:       exp_tile([[ZERO]]);

// Synchronize DST registers before pack
// CHECK-NEXT:       tile_regs_commit();
// CHECK-NEXT:       tile_regs_wait();

// Reserve output CB2
// CHECK-NEXT:       cb_reserve_back(get_compile_time_arg_val(2), [[TILES]]);

// Pack result to output CB2
// CHECK-NEXT:       pack_tile{{.*}}([[ZERO]], get_compile_time_arg_val(2), [[CB_IDX]]);

// Push to signal data ready
// CHECK-NEXT:       cb_push_back(get_compile_time_arg_val(2), [[TILES]]);

// Release DST registers (inside loop)
// CHECK-NEXT:       tile_regs_release();

// End loops
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return;

// Compute kernel: reads from CB0, CB1, computes f(A+B), writes to CB2
func.func @compute_fused(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                         %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>>
    attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %output = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>

  // Wait for inputs from reader thread
  %a_ready = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_ready = ttl.cb_wait %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %output_cb = ttl.attach_cb %output, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Fused computation: f(A + B) where f is exp
  %result = ttl.compute
      ins(%a_ready, %b_ready : tensor<2x2x!ttcore.tile<32x32, f32>>,
                               tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%output_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %c0 = arith.constant 0 : index
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %sum : !ttcore.tile<32x32, f32>
    %result_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %exp, %result_view[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// CHECK-LABEL: // writer_unary
// CHECK: void kernel_main() {
// CHECK-DAG:   size_t [[ONE:.*]] = 1;
// CHECK-DAG:   size_t [[BOUND:.*]] = 2;
// CHECK-DAG:   size_t [[PAGE_SIZE:.*]] = 4096;
// CHECK-DAG:   size_t [[ZERO:.*]] = 0;

// Write output to DRAM from CB2
// CHECK:   int32_t [[RT_ARG_OUT:.*]] = get_common_arg_val<uint32_t>([[ZERO]]);
// CHECK-NEXT:   auto [[ARGS_OUT:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<1, 0>();
// CHECK-NEXT:   TensorAccessor [[ACC_OUT:.*]] = TensorAccessor([[ARGS_OUT]], [[RT_ARG_OUT]],
// CHECK:   int32_t [[CB2_PTR:.*]] = get_read_ptr(get_compile_time_arg_val(2));
// Cast CB ptr to size_t for index arithmetic
// CHECK-NEXT:   ptrdiff_t [[CB2_PTR_PTRDIFF:v[0-9]+]] = (ptrdiff_t) [[CB2_PTR]];
// CHECK-NEXT:   size_t [[CB2_PTR_IDX:v[0-9]+]] = (size_t) [[CB2_PTR_PTRDIFF]];
// CHECK-NEXT:   for (size_t [[I_OUT:.*]] = [[ZERO]]; [[I_OUT]] < [[BOUND]]; [[I_OUT]] += [[ONE]]) {
// CHECK-NEXT:     for (size_t [[J_OUT:.*]] = [[ZERO]]; [[J_OUT]] < [[BOUND]]; [[J_OUT]] += [[ONE]]) {
// Tile offset computation: i * cols + j
// CHECK:       size_t [[TILE_OFF_OUT_Y:v[0-9]+]] = [[I_OUT]] * [[BOUND]];
// CHECK-NEXT:       size_t [[TILE_OFF_OUT_X:v[0-9]+]] = [[TILE_OFF_OUT_Y]] + [[J_OUT]];
// CB address computation: cb_ptr + tile_offset * page_size (all size_t arithmetic)
// CHECK-NEXT:       size_t [[BYTE_OFF_OUT:v[0-9]+]] = [[TILE_OFF_OUT_X]] * [[PAGE_SIZE]];
// CHECK-NEXT:       size_t [[CB_ADDR_OUT_IDX:v[0-9]+]] = [[CB2_PTR_IDX]] + [[BYTE_OFF_OUT]];
// Cast to i32 for NOC operation
// CHECK-NEXT:       ptrdiff_t [[TILE_OFF_OUT_PTR:v[0-9]+]] = (ptrdiff_t) [[TILE_OFF_OUT_X]];
// CHECK-NEXT:       int32_t [[TILE_OFF_OUT:v[0-9]+]] = (int32_t) [[TILE_OFF_OUT_PTR]];
// CHECK-NEXT:       ptrdiff_t [[CB_ADDR_OUT_PTR:v[0-9]+]] = (ptrdiff_t) [[CB_ADDR_OUT_IDX]];
// CHECK-NEXT:       int32_t [[CB_ADDR_OUT:v[0-9]+]] = (int32_t) [[CB_ADDR_OUT_PTR]];
// CHECK-NEXT:       noc_async_write_tile([[TILE_OFF_OUT]], [[ACC_OUT]], [[CB_ADDR_OUT]]);
// CHECK:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   noc_async_write_barrier();
// CHECK-NEXT:   return;
// CHECK-NEXT: }

// Writer kernel: pops from CB2, writes to DRAM
func.func @writer_unary(%out: tensor<64x64xf32, #layout>)
    attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>

  // Wait for data from compute thread (must match CB shape)
  %cb2_view = ttl.cb_wait %cb2 : <[2, 2], f32, 2> -> tensor<2x2xf32>

  // Copy from CB2 to output tensor
  %slice_out = ttl.tensor_slice %out[%c0, %c0] : tensor<64x64xf32, #layout> -> tensor<64x64xf32, #layout>
  %xf_out = ttl.copy %cb2, %slice_out : (!ttl.cb<[2, 2], f32, 2>, tensor<64x64xf32, #layout>) -> !ttl.transfer_handle<write>
  ttl.wait %xf_out : !ttl.transfer_handle<write>

  func.return
}
