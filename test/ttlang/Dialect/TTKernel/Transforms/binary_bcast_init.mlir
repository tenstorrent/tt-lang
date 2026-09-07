// Summary: ttkernel.binary_bcast gets a binary_bcast_init carrying both input
// CBs, the elementwise op and the broadcast dimension. The output CB is not
// part of the per-op init: PACK is configured once per sync region by
// binary_op_init_common. Consecutive ops sharing all of those reuse one init;
// changing either attribute forces a re-init. Because the common init only
// programs the data formats of the first input pair, an op reading another
// pair is preceded by a reconfig_data_format.

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
// The input formats are unchanged, so the re-init needs no reconfiguration.
// CHECK-LABEL: func.func @binary_bcast_reinit_on_op_change
// CHECK-NOT: ttkernel.reconfig_data_format
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

// -----

// Two input pairs in one sync region: the common init programs the formats of
// the first pair, so only the op reading the second pair reconfigures them.
// CHECK-LABEL: func.func @binary_bcast_reconfig_on_input_pair_change
// CHECK-DAG: %[[IN0:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[IN1:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-DAG: %[[IN2:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK-DAG: %[[IN3:.*]] = ttkernel.get_compile_time_arg_val(3)
// CHECK-DAG: %[[OUT:.*]] = ttkernel.get_compile_time_arg_val(4)
// CHECK: ttkernel.binary_op_init_common(%[[IN0]], %[[IN1]], %[[OUT]])
// CHECK-NOT: ttkernel.reconfig_data_format
// CHECK: ttkernel.binary_bcast_init(%[[IN0]], %[[IN1]], <mul>, <row>)
// CHECK-NEXT: ttkernel.binary_bcast(%[[IN0]], %[[IN1]],
// CHECK-NEXT: ttkernel.reconfig_data_format(%[[IN2]], %[[IN3]])
// CHECK-NEXT: ttkernel.binary_bcast_init(%[[IN2]], %[[IN3]], <mul>, <row>)
// CHECK-NEXT: ttkernel.binary_bcast(%[[IN2]], %[[IN3]],
func.func @binary_bcast_reconfig_on_input_pair_change() {
  %in0_cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %in1_cb = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %in2_cb = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, f32>>
  %in3_cb = ttkernel.get_compile_time_arg_val(3) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, f32>>
  %out_cb = ttkernel.get_compile_time_arg_val(4) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.binary_bcast(%in0_cb, %in1_cb, %c0, %c0, %c0, <mul>, <row>) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
  ttkernel.binary_bcast(%in2_cb, %in3_cb, %c0, %c0, %c1, <mul>, <row>) : (!ttkernel.cb<2, !ttcore.tile<32x32, f32>>, !ttkernel.cb<2, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %out_cb, %c0, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.pack_tile(%c1, %out_cb, %c1, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// -----

// Returning to the first pair reconfigures again: the formats now hold the
// second pair, so the third op cannot reuse the common init's configuration.
// CHECK-LABEL: func.func @binary_bcast_reconfig_back_to_first_pair
// CHECK-DAG: %[[IN0:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[IN1:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-DAG: %[[IN2:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK-DAG: %[[IN3:.*]] = ttkernel.get_compile_time_arg_val(3)
// CHECK: ttkernel.binary_bcast_init(%[[IN0]], %[[IN1]], <add>, <col>)
// CHECK-NEXT: ttkernel.binary_bcast(%[[IN0]], %[[IN1]],
// CHECK-NEXT: ttkernel.reconfig_data_format(%[[IN2]], %[[IN3]])
// CHECK-NEXT: ttkernel.binary_bcast_init(%[[IN2]], %[[IN3]], <add>, <col>)
// CHECK-NEXT: ttkernel.binary_bcast(%[[IN2]], %[[IN3]],
// CHECK-NEXT: ttkernel.reconfig_data_format(%[[IN0]], %[[IN1]])
// CHECK-NEXT: ttkernel.binary_bcast_init(%[[IN0]], %[[IN1]], <add>, <col>)
// CHECK-NEXT: ttkernel.binary_bcast(%[[IN0]], %[[IN1]],
func.func @binary_bcast_reconfig_back_to_first_pair() {
  %in0_cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %in1_cb = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %in2_cb = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, f32>>
  %in3_cb = ttkernel.get_compile_time_arg_val(3) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, f32>>
  %out_cb = ttkernel.get_compile_time_arg_val(4) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.binary_bcast(%in0_cb, %in1_cb, %c0, %c0, %c0, <add>, <col>) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
  ttkernel.binary_bcast(%in2_cb, %in3_cb, %c0, %c0, %c1, <add>, <col>) : (!ttkernel.cb<2, !ttcore.tile<32x32, f32>>, !ttkernel.cb<2, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  ttkernel.binary_bcast(%in0_cb, %in1_cb, %c1, %c0, %c0, <add>, <col>) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %out_cb, %c0, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.pack_tile(%c1, %out_cb, %c1, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}
