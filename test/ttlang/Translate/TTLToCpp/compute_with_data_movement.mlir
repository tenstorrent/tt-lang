// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-tile-and-assign-dst,ttl-insert-tile-regs-sync,ttl-lower-to-loops,ttl-annotate-cb-associations),convert-ttl-to-ttkernel,canonicalize,cse,lower-affine)' \
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
// CHECK-DAG:   size_t [[ZERO:.*]] = 0;

// Accessors materialized at function entry with chaining
// CHECK:   int32_t [[RT_ARG_A:.*]] = get_common_arg_val<uint32_t>([[ZERO]]);
// First accessor uses literal base CTA index = num_cbs = 2
// CHECK-NEXT:   auto [[ARGS_A:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<2, 0>();
// CHECK-NEXT:   TensorAccessor [[ACC_A:.*]] = TensorAccessor([[ARGS_A]], [[RT_ARG_A]],
// Second accessor chains from first
// CHECK:   int32_t [[RT_ARG_B:.*]] = get_common_arg_val<uint32_t>([[ONE]]);
// CHECK-NEXT:   auto [[ARGS_B:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<[[ARGS_A]].next_compile_time_args_offset(), [[ARGS_A]].next_common_runtime_args_offset()>();
// CHECK-NEXT:   TensorAccessor [[ACC_B:.*]] = TensorAccessor([[ARGS_B]], [[RT_ARG_B]],

// Read tensor A into CB0
// CHECK:   int32_t {{.*}} = get_common_arg_val<uint32_t>([[ZERO]]);
// CHECK:   int32_t [[CB0_PTR:.*]] = get_write_ptr(get_compile_time_arg_val(0));
// CHECK-NEXT:   for (size_t [[I_A:.*]] = [[ZERO]]; [[I_A]] < [[BOUND]]; [[I_A]] += [[ONE]]) {
// CHECK-NEXT:     for (size_t [[J_A:.*]] = [[ZERO]]; [[J_A]] < [[BOUND]]; [[J_A]] += [[ONE]]) {
// CHECK:       noc_async_read_tile({{.*}}, [[ACC_A]], [[CB0_PTR]]);
// CHECK:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   noc_async_read_barrier();

// Read tensor B into CB1
// CHECK:   int32_t {{.*}} = get_common_arg_val<uint32_t>([[ONE]]);
// CHECK:   int32_t [[CB1_PTR:.*]] = get_write_ptr(get_compile_time_arg_val(1));
// CHECK-NEXT:   for (size_t [[I_B:.*]] = [[ZERO]]; [[I_B]] < [[BOUND]]; [[I_B]] += [[ONE]]) {
// CHECK-NEXT:     for (size_t [[J_B:.*]] = [[ZERO]]; [[J_B]] < [[BOUND]]; [[J_B]] += [[ONE]]) {
// CHECK:       noc_async_read_tile({{.*}}, [[ACC_B]], [[CB1_PTR]]);
// CHECK:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   noc_async_read_barrier();
// CHECK-NEXT:   return;
// CHECK-NEXT: }

// Reader kernel: reads A and B from DRAM, pushes to CB0 and CB1
func.func @reader_binary(%a: tensor<64x64xf32, #layout>, %b: tensor<64x64xf32, #layout>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.base_cta_index = 2 : i32} {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>

  // Copy A to CB0
  %xf_a = ttl.copy %a, %cb0 : (tensor<64x64xf32, #layout>, !ttl.cb<[2, 2], f32, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %xf_a : !ttl.transfer_handle<read>

  // Copy B to CB1
  %xf_b = ttl.copy %b, %cb1 : (tensor<64x64xf32, #layout>, !ttl.cb<[2, 2], f32, 2>) -> !ttl.transfer_handle<read>
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

// Acquire DST registers
// CHECK-NEXT:   tile_regs_acquire();

// Nested loops over 2x2 tile grid
// CHECK-NEXT:   for (size_t [[I:.*]] = [[ZERO]]; [[I]] < [[BOUND]]; [[I]] += [[ONE]]) {
// CHECK-NEXT:     for (size_t [[J:.*]] = [[ZERO]]; [[J]] < [[BOUND]]; [[J]] += [[ONE]]) {

// Compute linear tile index: i * cols + j
// CHECK:            size_t [[COL_SIZE:.*]] = 2;
// CHECK-NEXT:       size_t [[IOFF:.*]] = [[I]] * [[COL_SIZE]];
// CHECK-NEXT:       size_t [[LINIDX:.*]] = [[IOFF]] + [[J]];

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
// CHECK-NEXT:       pack_tile{{.*}}([[ZERO]], get_compile_time_arg_val(2), [[ZERO]]);

// Push to signal data ready
// CHECK-NEXT:       cb_push_back(get_compile_time_arg_val(2), [[TILES]]);

// End loops
// CHECK-NEXT:     }
// CHECK-NEXT:   }

// Release DST registers
// CHECK-NEXT:   tile_regs_release();
// CHECK-NEXT:   return;

// Compute kernel: reads from CB0, CB1, computes f(A+B), writes to CB2
func.func @compute_fused(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                         %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
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
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %sum : !ttcore.tile<32x32, f32>
    %result_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %exp, %result_view : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// CHECK-LABEL: // writer_unary
// CHECK: void kernel_main() {
// CHECK-DAG:   size_t [[ONE:.*]] = 1;
// CHECK-DAG:   size_t [[BOUND:.*]] = 2;
// CHECK-DAG:   size_t [[ZERO:.*]] = 0;

// Write output to DRAM from CB2
// CHECK:   int32_t [[RT_ARG_OUT:.*]] = get_common_arg_val<uint32_t>([[ZERO]]);
// TensorAccessorArgs uses base CTA index = num_cbs = 1 (only cb2 bound in this func)
// CHECK-NEXT:   auto [[ARGS_OUT:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<1, 0>();
// CHECK-NEXT:   TensorAccessor [[ACC_OUT:.*]] = TensorAccessor([[ARGS_OUT]], [[RT_ARG_OUT]],
// CHECK:   int32_t [[CB2_PTR:.*]] = get_read_ptr(get_compile_time_arg_val(2));
// CHECK-NEXT:   for (size_t [[I_OUT:.*]] = [[ZERO]]; [[I_OUT]] < [[BOUND]]; [[I_OUT]] += [[ONE]]) {
// CHECK-NEXT:     for (size_t [[J_OUT:.*]] = [[ZERO]]; [[J_OUT]] < [[BOUND]]; [[J_OUT]] += [[ONE]]) {
// CHECK:       noc_async_write_tile({{.*}}, [[ACC_OUT]], [[CB2_PTR]]);
// CHECK:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   noc_async_write_barrier();
// CHECK-NEXT:   return;
// CHECK-NEXT: }

// Writer kernel: pops from CB2, writes to DRAM
func.func @writer_unary(%out: tensor<64x64xf32, #layout>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.base_cta_index = 1 : i32} {
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>

  // Wait for data from compute thread (must match CB shape)
  %cb2_view = ttl.cb_wait %cb2 : <[2, 2], f32, 2> -> tensor<2x2xf32>

  // Copy from CB2 to output tensor
  %xf_out = ttl.copy %cb2, %out : (!ttl.cb<[2, 2], f32, 2>, tensor<64x64xf32, #layout>) -> !ttl.transfer_handle<write>
  ttl.wait %xf_out : !ttl.transfer_handle<write>

  func.return
}
