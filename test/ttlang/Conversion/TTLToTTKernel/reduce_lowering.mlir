// Summary: Tests for reduce tile op lowering to TTKernel.
// Input is pre-lowered IR (after convert-ttl-to-compute, assign-dst,
// insert-tile-regs-sync, lower-to-loops, annotate-cb-associations).
// Tests only the TTKernel conversion, init insertion, and L1 accumulation.
//
// Full-fp32 accumulation follows the kernel's fp32_dest_acc_en and does not
// change this lowering; target-specific selection of it is covered by
// test/ttlang/Dialect/TTL/Transforms/SetComputeKernelConfig/
// set_compute_kernel_config.mlir.

// RUN: ttlang-opt %s --split-input-file \
// RUN:   -pass-pipeline='builtin.module( \
// RUN:     convert-ttl-to-ttkernel, \
// RUN:     ttkernel-insert-inits, ttkernel-insert-l1-accumulation, \
// RUN:     canonicalize, cse)' \
// RUN:   | FileCheck %s

// Single-tile reduce_sum along dim 0 (REDUCE_COL).
// Verifies the reduce_init -> reduce_tile -> reduce_uninit sequence and CB
// routing.
// CHECK-LABEL: func.func @reduce_sum_dim0_1x1
// CHECK-DAG: %[[C1I:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: ttkernel.init_sfpu(%[[CB0]], %[[CB2]])
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.reduce_init(%[[CB0]], %[[CB1]], %[[CB2]], <reduce_sum>, <reduce_dim_col>)
// CHECK-NEXT: ttkernel.reduce_tile(%[[CB0]], %[[CB1]], %[[C0]], %[[C0]], %[[C0]], <reduce_sum>, <reduce_dim_col>)
// CHECK: ttkernel.reduce_uninit()
// CHECK: ttkernel.pack_tile(%[[C0]], %[[CB2]], %[[C0]], true)
func.func @reduce_sum_dim0_1x1() attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>, fp32_dest_acc_en = true} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %inp = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %inp_cb = ttl.attach_cb %inp, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler_cb = ttl.attach_cb %scaler, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_cb = ttl.attach_cb %empty, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.for %iv0 = %c0 to %c1 step %c1 {
    scf.for %iv1 = %c0 to %c1 step %c1 {
      %in_tile = tensor.extract %inp_cb[%iv0, %iv1] : tensor<1x1x!ttcore.tile<32x32, bf16>>
      %sc_tile = tensor.extract %scaler_cb[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
      %out_tile = tensor.extract %out_cb[%c0, %iv1] : tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.tile_regs_acquire
      %red = ttl.tile_reduce %in_tile, %sc_tile, %out_tile 0 : i32 <reduce_dim_col> into dst[%c0] {ttl.reduce_output_cb_index = 2 : index} : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      ttl.tile_regs_commit
      ttl.tile_regs_wait
      ttl.tile_store %red, %reserve[%c0, %iv1] from dst[%c0] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.tile_regs_release
    } {ttl.tile_loop_stride = 1 : index}
  } {ttl.reduction_loop, ttl.tile_loop_stride = 1 : index}
  ttl.cb_push %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Single-tile reduce_sum along dim 1 (REDUCE_ROW).
// CHECK-LABEL: func.func @reduce_sum_dim1_1x1
// CHECK: ttkernel.reduce_init({{.*}}<reduce_sum>, <reduce_dim_row>)
// CHECK: ttkernel.reduce_tile({{.*}}<reduce_sum>, <reduce_dim_row>)
// CHECK: ttkernel.reduce_uninit()
// CHECK: ttkernel.pack_tile
func.func @reduce_sum_dim1_1x1() attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>, fp32_dest_acc_en = true} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %inp = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %inp_cb = ttl.attach_cb %inp, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler_cb = ttl.attach_cb %scaler, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_cb = ttl.attach_cb %empty, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.for %iv0 = %c0 to %c1 step %c1 {
    scf.for %iv1 = %c0 to %c1 step %c1 {
      %in_tile = tensor.extract %inp_cb[%iv0, %iv1] : tensor<1x1x!ttcore.tile<32x32, bf16>>
      %sc_tile = tensor.extract %scaler_cb[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
      %out_tile = tensor.extract %out_cb[%iv0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.tile_regs_acquire
      %red = ttl.tile_reduce %in_tile, %sc_tile, %out_tile 0 : i32 <reduce_dim_row> into dst[%c0] {ttl.reduce_output_cb_index = 2 : index} : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      ttl.tile_regs_commit
      ttl.tile_regs_wait
      ttl.tile_store %red, %reserve[%iv0, %c0] from dst[%c0] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.tile_regs_release
    } {ttl.tile_loop_stride = 1 : index}
  } {ttl.reduction_loop, ttl.tile_loop_stride = 1 : index}
  ttl.cb_push %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Multi-tile reduce (2x1 -> 1x1): reduction loop with L1 accumulation guard.
// CHECK-LABEL: func.func @reduce_2x1_l1_acc
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C0I:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C1I:.*]] = arith.constant 1 : i32
// CHECK: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// Disable L1 accumulation before the reduction loop.
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[C0I]])
// CHECK: scf.for %[[IV:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-NEXT:   ttkernel.tile_regs_acquire
// CHECK:   ttkernel.reduce_init({{.*}}<reduce_sum>, <reduce_dim_col>)
// CHECK:   ttkernel.reduce_tile({{.*}}<reduce_sum>, <reduce_dim_col>)
// CHECK:   ttkernel.reduce_uninit()
// CHECK:   ttkernel.pack_tile(%[[C0]], %[[CB2]], %[[C0]], true)
// CHECK:   ttkernel.tile_regs_release
// L1 accumulation guard: enable once after the first iteration's pack.
// CHECK:   %[[FIRST:.*]] = arith.cmpi eq, %[[IV]], %[[C0]]
// CHECK-NEXT:   scf.if %[[FIRST]]
// CHECK-NEXT:     ttkernel.pack_reconfig_l1_acc(%[[C1I]])
// CHECK:        }
// CHECK: } {ttl.reduction_loop
// Disable L1 accumulation after reduction loop.
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[C0I]])
func.func @reduce_2x1_l1_acc() attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>, fp32_dest_acc_en = true} {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %inp = ttl.cb_wait %cb0 : <[2, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %inp_cb = ttl.attach_cb %inp, %cb0 : (tensor<2x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler_cb = ttl.attach_cb %scaler, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_cb = ttl.attach_cb %empty, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.for %iv0 = %c0 to %c2 step %c1 {
    scf.for %iv1 = %c0 to %c1 step %c1 {
      %in_tile = tensor.extract %inp_cb[%iv0, %iv1] : tensor<2x1x!ttcore.tile<32x32, bf16>>
      %sc_tile = tensor.extract %scaler_cb[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
      %out_tile = tensor.extract %out_cb[%c0, %iv1] : tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.tile_regs_acquire
      %red = ttl.tile_reduce %in_tile, %sc_tile, %out_tile 0 : i32 <reduce_dim_col> into dst[%c0] {ttl.reduce_output_cb_index = 2 : index} : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      ttl.tile_regs_commit
      ttl.tile_regs_wait
      ttl.tile_store %red, %reserve[%c0, %iv1] from dst[%c0] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.tile_regs_release
    } {ttl.tile_loop_stride = 1 : index}
  } {ttl.reduction_loop, ttl.tile_loop_stride = 1 : index}
  ttl.cb_push %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : <[2, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}
