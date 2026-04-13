// Summary: Tests for reduce tile op lowering to TTKernel.
// Input is pre-lowered IR (after convert-ttl-to-compute, assign-dst,
// insert-tile-regs-sync, lower-to-loops, annotate-cb-associations).
// Tests only the TTKernel conversion, init insertion, and L1 accumulation.

// full_fp32 enabled:
// RUN: ttlang-opt %s --split-input-file \
// RUN:   -pass-pipeline='builtin.module( \
// RUN:     convert-ttl-to-ttkernel{reduce-full-fp32=true}, \
// RUN:     ttkernel-insert-inits, ttkernel-insert-l1-accumulation, \
// RUN:     canonicalize, cse)' \
// RUN:   | FileCheck %s --check-prefix=FP32

// full_fp32 disabled:
// RUN: ttlang-opt %s --split-input-file \
// RUN:   -pass-pipeline='builtin.module( \
// RUN:     convert-ttl-to-ttkernel{reduce-full-fp32=false}, \
// RUN:     ttkernel-insert-inits, ttkernel-insert-l1-accumulation, \
// RUN:     canonicalize, cse)' \
// RUN:   | FileCheck %s --check-prefix=NOFP32

// Single-tile reduce_sum along dim 0 (REDUCE_COL).
// Verifies: reduce_init -> reduce_tile -> reduce_uninit sequence,
// correct CB routing, and full_fp32 attribute presence/absence.
// FP32-LABEL: func.func @reduce_sum_dim0_1x1
// FP32-DAG: %[[C1I:.*]] = arith.constant 1 : i32
// FP32-DAG: %[[C0:.*]] = arith.constant 0 : index
// FP32: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// FP32: %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// FP32: %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// FP32: ttkernel.init_sfpu(%[[CB0]], %[[CB2]])
// FP32: ttkernel.tile_regs_acquire
// FP32: ttkernel.reduce_init(%[[CB0]], %[[CB1]], %[[CB2]], <reduce_sum>, <reduce_dim_col>) {full_fp32}
// FP32-NEXT: ttkernel.reduce_tile(%[[CB0]], %[[CB1]], %[[C0]], %[[C0]], %[[C0]], <reduce_sum>, <reduce_dim_col>) {full_fp32
// FP32: ttkernel.reduce_uninit
// FP32: ttkernel.pack_tile(%[[C0]], %[[CB2]], %[[C0]], true)
//
// NOFP32-LABEL: func.func @reduce_sum_dim0_1x1
// NOFP32: ttkernel.tile_regs_acquire
// NOFP32: ttkernel.reduce_init({{.*}}<reduce_sum>, <reduce_dim_col>)
// NOFP32-NOT: full_fp32
// NOFP32: ttkernel.reduce_tile({{.*}}<reduce_sum>, <reduce_dim_col>)
// NOFP32-NOT: full_fp32
// NOFP32: ttkernel.reduce_uninit
func.func @reduce_sum_dim0_1x1() attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
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

// Multi-tile reduce (2x1 -> 1x1): reduction loop with L1 accumulation guard.
// FP32-LABEL: func.func @reduce_2x1_l1_acc
// FP32-DAG: %[[C0:.*]] = arith.constant 0 : index
// FP32-DAG: %[[C1:.*]] = arith.constant 1 : index
// FP32-DAG: %[[C2:.*]] = arith.constant 2 : index
// FP32-DAG: %[[C0I:.*]] = arith.constant 0 : i32
// FP32-DAG: %[[C1I:.*]] = arith.constant 1 : i32
// FP32: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// FP32: %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// FP32: %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2)
// Disable L1 accumulation before the reduction loop.
// FP32: ttkernel.pack_reconfig_l1_acc(%[[C0I]])
// FP32: scf.for %[[IV:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
// FP32-NEXT:   ttkernel.tile_regs_acquire
// FP32:   ttkernel.reduce_init({{.*}}<reduce_sum>, <reduce_dim_col>) {full_fp32}
// FP32:   ttkernel.reduce_tile({{.*}}<reduce_sum>, <reduce_dim_col>) {full_fp32
// FP32:   ttkernel.reduce_uninit
// FP32:   ttkernel.pack_tile(%[[C0]], %[[CB2]], %[[C0]], true)
// FP32:   ttkernel.tile_regs_release
// L1 accumulation guard: enable once after the first iteration's pack.
// FP32:   %[[FIRST:.*]] = arith.cmpi eq, %[[IV]], %[[C0]]
// FP32-NEXT:   scf.if %[[FIRST]]
// FP32-NEXT:     ttkernel.pack_reconfig_l1_acc(%[[C1I]])
// FP32:        }
// FP32: } {ttl.reduction_loop
// Disable L1 accumulation after reduction loop.
// FP32: ttkernel.pack_reconfig_l1_acc(%[[C0I]])
func.func @reduce_2x1_l1_acc() attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
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
