// RUN: ttlang-opt %s --ttkernel-insert-inits | FileCheck %s

// Summary: a destination-register section may pack two element types. The
// common init configures the first pack format, and a later pack of a
// different format is preceded by pack_reconfig_data_format.

// CHECK-LABEL: func.func @multiple_output_cbs_different_formats
// CHECK: ttkernel.init_sfpu
// CHECK: ttkernel.pack_tile
// CHECK: ttkernel.pack_reconfig_data_format(%[[OUT:.*]]) : (!ttkernel.cb<2, !ttcore.tile<32x32, f32>>)
// CHECK-NEXT: ttkernel.pack_tile({{.*}}%[[OUT]]
func.func @multiple_output_cbs_different_formats() {
  %cb_bf16 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %cb_f32 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%cb_bf16, %c0, %c0) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.exp_tile(%c0) : (index) -> ()
  ttkernel.pack_tile(%c0, %cb_bf16, %c0, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.pack_tile(%c0, %cb_f32, %c0, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, f32>>, index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// transpose_wh_init configures the packer for its output CB. The following
// pack targets a different element type, so it is reconfigured even though
// it is the only pack in the section.
// CHECK-LABEL: func.func @transpose_then_pack_different_format
// CHECK: ttkernel.transpose_wh_init
// CHECK: ttkernel.transpose_wh_init(%[[IDX:.*]], %[[IDX]])
// CHECK: ttkernel.pack_reconfig_data_format(%[[KEYS:.*]]) : (!ttkernel.cb<2, !ttcore.tile<32x32, u32>>)
// CHECK-NEXT: ttkernel.pack_tile({{.*}}%[[KEYS]]
func.func @transpose_then_pack_different_format() {
  %values = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %indices = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, u16>>
  %keys = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, u32>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.transpose_wh_tile(%values, %c0, %c0) {ttl.transpose_output_cb_index = 2 : index} : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.transpose_wh_tile(%indices, %c0, %c1) {ttl.transpose_output_cb_index = 1 : index} : (!ttkernel.cb<2, !ttcore.tile<32x32, u16>>, index, index) -> ()
  ttkernel.pack_tile(%c0, %keys, %c0, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, u32>>, index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}
