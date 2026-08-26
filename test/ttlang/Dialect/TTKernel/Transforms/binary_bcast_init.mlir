// Summary: ttkernel.binary_bcast gets a binary_bcast_init carrying both input
// CBs, the elementwise op and the broadcast dimension. The output CB is not
// part of the per-op init: PACK is configured once per sync region by
// binary_op_init_common. Consecutive ops sharing all of those reuse one init;
// changing either attribute forces a re-init.

// RUN: ttlang-opt %s --ttkernel-insert-inits | FileCheck %s

// CHECK-LABEL: func.func @binary_bcast_single
// CHECK-DAG: %[[IN0:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[IN1:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-DAG: %[[OUT:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: ttkernel.binary_op_init_common(%[[IN0]], %[[IN1]], %[[OUT]])
// CHECK: ttkernel.binary_bcast_init(%[[IN0]], %[[IN1]], <add>, <col>)
// CHECK-NEXT: ttkernel.binary_bcast(%[[IN0]], %[[IN1]],
func.func @binary_bcast_single() {
  %in0_cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %in1_cb = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %out_cb = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.binary_bcast(%in0_cb, %in1_cb, %c0, %c0, %c0, <add>, <col>) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %out_cb, %c0, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// -----

// Two ops with identical op/bcast kinds and CBs share one init.
// CHECK-LABEL: func.func @binary_bcast_shared_init
// CHECK: ttkernel.binary_bcast_init({{.*}}, <mul>, <row>)
// CHECK-NEXT: ttkernel.binary_bcast(
// CHECK-NEXT: ttkernel.binary_bcast(
// CHECK-NOT: ttkernel.binary_bcast_init
func.func @binary_bcast_shared_init() {
  %in0_cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %in1_cb = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %out_cb = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.binary_bcast(%in0_cb, %in1_cb, %c0, %c0, %c0, <mul>, <row>) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
  ttkernel.binary_bcast(%in0_cb, %in1_cb, %c1, %c0, %c1, <mul>, <row>) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %out_cb, %c0, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// -----

// Changing the elementwise op re-inits even though the CBs and bcast dim match.
// CHECK-LABEL: func.func @binary_bcast_reinit_on_op_change
// CHECK: ttkernel.binary_bcast_init({{.*}}, <add>, <scalar>)
// CHECK-NEXT: ttkernel.binary_bcast(
// CHECK-NEXT: ttkernel.binary_bcast_init({{.*}}, <sub>, <scalar>)
// CHECK-NEXT: ttkernel.binary_bcast(
func.func @binary_bcast_reinit_on_op_change() {
  %in0_cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %in1_cb = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %out_cb = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.binary_bcast(%in0_cb, %in1_cb, %c0, %c0, %c0, <add>, <scalar>) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
  ttkernel.binary_bcast(%in0_cb, %in1_cb, %c1, %c0, %c1, <sub>, <scalar>) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %out_cb, %c0, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}
