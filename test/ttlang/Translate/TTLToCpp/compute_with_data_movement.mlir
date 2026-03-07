// FPU path (default): add uses add_tiles (reads from CB), no copy_tile for add.
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-assign-dst,ttl-insert-tile-regs-sync,ttl-lower-to-loops,ttl-annotate-cb-associations),convert-ttl-to-ttkernel,ttkernel-insert-inits,canonicalize,cse,lower-affine)' \
// RUN:   -o %t.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=FPU

// SFPU path: all binary ops use copy_tile + SFPU binary ops.
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-assign-dst{enable-fpu-binary-ops=0},ttl-insert-tile-regs-sync,ttl-lower-to-loops,ttl-annotate-cb-associations),convert-ttl-to-ttkernel,ttkernel-insert-inits,canonicalize,cse,lower-affine)' \
// RUN:   -o %t.sfpu.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.sfpu.ttkernel.mlir -o %t.sfpu.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.sfpu.cpp %t.sfpu.emitc.mlir
// RUN: FileCheck %s --input-file=%t.sfpu.cpp --check-prefix=SFPU

// Purpose: Complete example with reader, compute, and writer threads.
// Pattern: reader (NOC) -> CBs -> compute (MATH) -> CB -> writer (NOC)
// Operation: f(A + B) where f is exp, matching the C++ example pattern.

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// FPU path: reader kernel (same for both paths)
// =============================================================================
// FPU-LABEL: // reader_binary
// FPU: void kernel_main() {
// FPU:   noc_async_read_tile(
// FPU:   noc_async_read_barrier();
// FPU:   noc_async_read_tile(
// FPU:   noc_async_read_barrier();
// FPU-NEXT:   return;

// =============================================================================
// FPU path: compute kernel — binary_op_init_common, add_tiles, exp
// =============================================================================
// FPU-LABEL: // compute_fused
// FPU: void kernel_main() {
// FPU-DAG:   int32_t [[TILES:v[0-9]+]] = 4
// FPU-DAG:   size_t [[ZERO:v[0-9]+]] = 0

// FPU:       cb_wait_front(get_compile_time_arg_val(0), [[TILES]]);
// FPU-NEXT:  cb_wait_front(get_compile_time_arg_val(1), [[TILES]]);
// FPU-NEXT:  cb_reserve_back(get_compile_time_arg_val(2), [[TILES]]);
// FPU-NEXT:  binary_op_init_common(get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(2));

// FPU:       tile_regs_acquire();
// No copy_tile for FPU add — operands read directly from CB
// FPU-NOT:   copy_tile
// FPU:       add_tiles_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1));
// FPU-NEXT:  add_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1),
// FPU:       exp_tile_init();
// FPU-NEXT:  exp_tile(
// FPU:       tile_regs_commit();
// FPU-NEXT:  tile_regs_wait();
// FPU:       pack_tile<true>([[ZERO]], get_compile_time_arg_val(2),
// FPU:       tile_regs_release();

// FPU-NOT:   init_sfpu
// FPU-NOT:   add_binary_tile

// FPU-LABEL: // writer_unary
// FPU:       noc_async_write_tile(
// FPU:       noc_async_write_barrier();

// =============================================================================
// SFPU path: reader kernel (same for both paths)
// =============================================================================
// SFPU-LABEL: // reader_binary
// SFPU: void kernel_main() {
// SFPU:   noc_async_read_tile(
// SFPU:   noc_async_read_barrier();
// SFPU:   noc_async_read_tile(
// SFPU:   noc_async_read_barrier();
// SFPU-NEXT:   return;

// =============================================================================
// SFPU path: compute kernel — init_sfpu, copy_tile, add_binary_tile, exp
// =============================================================================
// SFPU-LABEL: // compute_fused
// SFPU: void kernel_main() {
// SFPU-DAG:   int32_t [[TILES:v[0-9]+]] = 4
// SFPU-DAG:   size_t [[ZERO:v[0-9]+]] = 0

// SFPU:       cb_wait_front(get_compile_time_arg_val(0), [[TILES]]);
// SFPU-NEXT:  cb_wait_front(get_compile_time_arg_val(1), [[TILES]]);
// SFPU-NEXT:  cb_reserve_back(get_compile_time_arg_val(2), [[TILES]]);
// SFPU-NEXT:  init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(2));

// SFPU:       tile_regs_acquire();
// SFPU:       copy_tile_init(get_compile_time_arg_val(0));
// SFPU-NEXT:  copy_tile(get_compile_time_arg_val(0), {{.*}}, [[ZERO]]);
// SFPU:       copy_tile_init(get_compile_time_arg_val(1));
// SFPU-NEXT:  copy_tile(get_compile_time_arg_val(1),
// SFPU:       add_binary_tile_init();
// SFPU-NEXT:  add_binary_tile(
// SFPU:       exp_tile_init();
// SFPU-NEXT:  exp_tile(
// SFPU:       tile_regs_commit();
// SFPU-NEXT:  tile_regs_wait();
// SFPU:       pack_tile<true>([[ZERO]], get_compile_time_arg_val(2),
// SFPU:       tile_regs_release();

// SFPU-NOT:   binary_op_init_common
// SFPU-NOT:   add_tiles

// SFPU-LABEL: // writer_unary
// SFPU:       noc_async_write_tile(
// SFPU:       noc_async_write_barrier();

// Reader kernel: reads A and B from DRAM, pushes to CB0 and CB1
func.func @reader_binary(%a: tensor<2x2x!ttcore.tile<32x32, f32>, #layout>, %b: tensor<2x2x!ttcore.tile<32x32, f32>, #layout>)
    attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [0, 1], ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>

  // Copy A to CB0
  %slice_a = ttl.tensor_slice %a[%c0, %c0] : tensor<2x2x!ttcore.tile<32x32, f32>, #layout> -> tensor<2x2x!ttcore.tile<32x32, f32>, #layout>
  %xf_a = ttl.copy %slice_a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[2, 2], f32, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %xf_a : !ttl.transfer_handle<read>

  // Copy B to CB1
  %slice_b = ttl.tensor_slice %b[%c0, %c0] : tensor<2x2x!ttcore.tile<32x32, f32>, #layout> -> tensor<2x2x!ttcore.tile<32x32, f32>, #layout>
  %xf_b = ttl.copy %slice_b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[2, 2], f32, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %xf_b : !ttl.transfer_handle<read>

  func.return
}

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
  %result_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
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
    ttl.tile_store %exp, %result_view : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// Writer kernel: pops from CB2, writes to DRAM
func.func @writer_unary(%out: tensor<2x2x!ttcore.tile<32x32, f32>, #layout>)
    attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>

  // Wait for data from compute thread (must match CB shape)
  %cb2_view = ttl.cb_wait %cb2 : <[2, 2], f32, 2> -> tensor<2x2xf32>

  // Copy from CB2 to output tensor
  %slice_out = ttl.tensor_slice %out[%c0, %c0] : tensor<2x2x!ttcore.tile<32x32, f32>, #layout> -> tensor<2x2x!ttcore.tile<32x32, f32>, #layout>
  %xf_out = ttl.copy %cb2, %slice_out : (!ttl.cb<[2, 2], f32, 2>, tensor<2x2x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
  ttl.wait %xf_out : !ttl.transfer_handle<write>

  func.return
}
