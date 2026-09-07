// Summary: Broadcast per-op inits reconfigure input formats without resetting
// the DST synchronization bank. Equal consecutive keys share an init; changing
// either input or hardware kind requires another init.

// RUN: ttlang-opt %s --split-input-file --ttkernel-insert-inits | FileCheck %s --implicit-check-not=ttkernel.compute_kernel_hw_startup

// CHECK-LABEL: func.func @binary_bcast_single
// CHECK-DAG: %[[IN0:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[IN1:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-DAG: %[[OUT:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: ttkernel.reconfig_data_format(%[[IN0]], %[[IN1]])
// CHECK-NEXT: ttkernel.binary_bcast_init(%[[IN0]], %[[IN1]], <add>, <col>)
// CHECK-NEXT: ttkernel.binary_bcast(%[[IN0]], %[[IN1]],
func.func @binary_bcast_single() {
  %in0_cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %in1_cb = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %out_cb = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.binary_bcast(%in0_cb, %in1_cb, %c0, %c0, %c0, <add>, <col>) {ttl.bcast_output_cb_index = 2 : index} : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %out_cb, %c0, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// -----

// Two ops with identical op/bcast kinds and CBs share one init.
// CHECK-LABEL: func.func @binary_bcast_shared_init
// CHECK: ttkernel.reconfig_data_format(
// CHECK-NEXT: ttkernel.binary_bcast_init({{.*}}, <mul>, <row>)
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
  ttkernel.binary_bcast(%in0_cb, %in1_cb, %c0, %c0, %c0, <mul>, <row>) {ttl.bcast_output_cb_index = 2 : index} : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
  ttkernel.binary_bcast(%in0_cb, %in1_cb, %c1, %c0, %c1, <mul>, <row>) {ttl.bcast_output_cb_index = 2 : index} : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
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
// CHECK: ttkernel.binary_bcast_init({{.*}}, <sub>, <scalar>)
// CHECK-NEXT: ttkernel.binary_bcast(
func.func @binary_bcast_reinit_on_op_change() {
  %in0_cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %in1_cb = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %out_cb = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.binary_bcast(%in0_cb, %in1_cb, %c0, %c0, %c0, <add>, <scalar>) {ttl.bcast_output_cb_index = 2 : index} : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
  ttkernel.binary_bcast(%in0_cb, %in1_cb, %c1, %c0, %c1, <sub>, <scalar>) {ttl.bcast_output_cb_index = 2 : index} : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %out_cb, %c0, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// -----

// Two input pairs in one sync region: each op configures the hardware for the
// pair it reads, so neither runs with the other pair's configuration. The
// region-level binary_op_init_common only covers the first pair, and a region
// like this one can sit inside a loop, where the configuration left behind is
// the one of the last op of the previous iteration.
// CHECK-LABEL: func.func @binary_bcast_two_input_pairs
// CHECK-DAG: %[[IN0:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[IN1:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-DAG: %[[IN2:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK-DAG: %[[IN3:.*]] = ttkernel.get_compile_time_arg_val(3)
// CHECK-DAG: %[[OUT:.*]] = ttkernel.get_compile_time_arg_val(4)
// CHECK: ttkernel.reconfig_data_format(%[[IN0]], %[[IN1]])
// CHECK-NEXT: ttkernel.binary_bcast_init(%[[IN0]], %[[IN1]], <mul>, <row>)
// CHECK-NEXT: ttkernel.binary_bcast(%[[IN0]], %[[IN1]],
// CHECK-NEXT: ttkernel.reconfig_data_format(%[[IN2]], %[[IN3]])
// CHECK-NEXT: ttkernel.binary_bcast_init(%[[IN2]], %[[IN3]], <mul>, <row>)
// CHECK-NEXT: ttkernel.binary_bcast(%[[IN2]], %[[IN3]],
// CHECK-NEXT: ttkernel.reconfig_data_format(%[[IN0]], %[[IN1]])
// CHECK-NEXT: ttkernel.binary_bcast_init(%[[IN0]], %[[IN1]], <mul>, <row>)
// CHECK-NEXT: ttkernel.binary_bcast(%[[IN0]], %[[IN1]],
func.func @binary_bcast_two_input_pairs() {
  %in0_cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %in1_cb = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %in2_cb = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, f32>>
  %in3_cb = ttkernel.get_compile_time_arg_val(3) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, f32>>
  %out_cb = ttkernel.get_compile_time_arg_val(4) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.binary_bcast(%in0_cb, %in1_cb, %c0, %c0, %c0, <mul>, <row>) {ttl.bcast_output_cb_index = 4 : index} : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
  ttkernel.binary_bcast(%in2_cb, %in3_cb, %c0, %c0, %c1, <mul>, <row>) {ttl.bcast_output_cb_index = 4 : index} : (!ttkernel.cb<2, !ttcore.tile<32x32, f32>>, !ttkernel.cb<2, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  ttkernel.binary_bcast(%in0_cb, %in1_cb, %c1, %c0, %c0, <mul>, <row>) {ttl.bcast_output_cb_index = 4 : index} : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %out_cb, %c0, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.pack_tile(%c1, %out_cb, %c1, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// -----

// Unary broadcast after a binary broadcast must not reset the active DST bank.
// Both input pairs are re-established on every iteration of the tile loop.
// CHECK-LABEL: func.func @binary_then_unary_bcast_in_loop
// CHECK-DAG: %[[IN0:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[IN1:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-DAG: %[[IN2:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: ttkernel.binary_op_init_common
// CHECK: scf.for
// CHECK-NEXT: ttkernel.tile_regs_acquire
// CHECK-NEXT: ttkernel.reconfig_data_format(%[[IN0]], %[[IN1]])
// CHECK-NEXT: ttkernel.binary_bcast_init
// CHECK-NEXT: ttkernel.binary_bcast
// CHECK: ttkernel.reconfig_data_format(%[[IN2]], %[[IN2]])
// CHECK-NEXT: ttkernel.unary_bcast_init(%[[IN2]], <col>)
// CHECK-NEXT: ttkernel.unary_bcast
// CHECK-NEXT: ttkernel.tile_regs_commit
func.func @binary_then_unary_bcast_in_loop() {
  %in0_cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %in1_cb = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %in2_cb = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, f32>>
  %in3_cb = ttkernel.get_compile_time_arg_val(3) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, f32>>
  %out_cb = ttkernel.get_compile_time_arg_val(4) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.binary_bcast(%in0_cb, %in1_cb, %c0, %c0, %c0, <mul>, <row>) {ttl.bcast_output_cb_index = 4 : index} : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
    ttkernel.binary_bcast(%in2_cb, %in3_cb, %c0, %c0, %c1, <mul>, <row>) {ttl.bcast_output_cb_index = 4 : index} : (!ttkernel.cb<2, !ttcore.tile<32x32, f32>>, !ttkernel.cb<2, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
    ttkernel.binary_bcast(%in0_cb, %in1_cb, %c1, %c0, %c0, <mul>, <row>) {ttl.bcast_output_cb_index = 4 : index} : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
    ttkernel.unary_bcast(%in2_cb, %c0, %c1, <col>) : (!ttkernel.cb<2, !ttcore.tile<32x32, f32>>, index, index) -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %out_cb, %c0, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.pack_tile(%c1, %out_cb, %c1, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.tile_loop_stride = 1 : i64}
  func.return
}
