// Summary: Test for transpose tile op lowering to TTKernel.
// Input is pre-lowered IR (after convert-ttl-to-compute, assign-dst,
// subblock, insert-tile-regs-sync, lower-to-loops, annotate-cb).
// Tests only TTKernel conversion and init insertion.

// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module( \
// RUN:     convert-ttl-to-ttkernel, ttkernel-insert-inits, \
// RUN:     ttkernel-insert-l1-accumulation, canonicalize, cse)' \
// RUN:   | FileCheck %s

// Single-tile transpose: transpose_wh_init -> transpose_wh_tile.
// CHECK-LABEL: func.func @transpose_1x1
// CHECK-DAG: %[[C1I:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.init_sfpu(%[[CB0]], %[[CB1]])
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.transpose_wh_init(%[[CB0]], %[[CB1]])
// CHECK-NEXT: ttkernel.transpose_wh_tile(%[[CB0]], %[[C0]], %[[C0]])
// CHECK: ttkernel.tile_regs_commit
// CHECK: ttkernel.tile_regs_wait
// CHECK: ttkernel.pack_tile(%[[C0]], %[[CB1]], %[[C0]], true)
// CHECK: ttkernel.tile_regs_release
func.func @transpose_1x1() attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %inp = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %inp_cb = ttl.attach_cb %inp, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_cb = ttl.attach_cb %empty, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.for %iv0 = %c0 to %c1 step %c1 {
    scf.for %iv1 = %c0 to %c1 step %c1 {
      %in_tile = tensor.extract %inp_cb[%iv1, %iv0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
      %out_tile = tensor.extract %out_cb[%iv0, %iv1] : tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.tile_regs_acquire
      %tr = ttl.tile_transpose %in_tile, %out_tile {dst_idx = 0 : i32, ttl.transpose_output_cb_index = 1 : index} : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      ttl.tile_regs_commit
      ttl.tile_regs_wait
      ttl.tile_store %tr, %reserve[%iv0, %iv1] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.tile_regs_release
    } {ttl.tile_loop_stride = 1 : index}
  } {ttl.tile_loop_stride = 1 : index}
  ttl.cb_push %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}
