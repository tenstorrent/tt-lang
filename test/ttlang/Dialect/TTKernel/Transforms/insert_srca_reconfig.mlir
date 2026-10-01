// copy_tile_init does not program the SrcA data format. The common init does,
// once, from the first input CB. A later copy from a different element type
// must reconfigure SrcA or the unpacker keeps the first format.
// RUN: ttlang-opt %s --ttkernel-insert-inits | FileCheck %s

// CHECK-LABEL: func.func @bf16_then_u16
// CHECK-DAG: %[[BF16:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[U16:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-DAG: %[[OUT:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: ttkernel.init_sfpu(%[[BF16]], %[[OUT]])
// CHECK-NEXT: ttkernel.tile_regs_acquire
// CHECK-NOT: ttkernel.reconfig_data_format_srca
// CHECK: ttkernel.copy_tile_init(%[[BF16]])
// CHECK-NEXT: ttkernel.copy_tile(%[[BF16]]
// CHECK-NEXT: ttkernel.reconfig_data_format_srca(%[[BF16]], %[[U16]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[U16]])
// CHECK-NEXT: ttkernel.copy_tile(%[[U16]]
// CHECK: ttkernel.tile_regs_release
func.func @bf16_then_u16() {
  %bf16 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %u16 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, u16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%bf16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.copy_tile(%u16, %c0, %c1) : (!ttkernel.cb<4, !ttcore.tile<32x32, u16>>, index, index) -> ()
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// Same element type, different CBs: copy_tile_init switches the operand and
// the unpack format stays valid.
// CHECK-LABEL: func.func @same_format_different_cbs
// CHECK: ttkernel.copy_tile_init
// CHECK-NEXT: ttkernel.copy_tile
// CHECK-NEXT: ttkernel.copy_tile_init
// CHECK-NEXT: ttkernel.copy_tile
// CHECK-NOT: ttkernel.reconfig_data_format_srca
// CHECK: ttkernel.tile_regs_release
func.func @same_format_different_cbs() {
  %lhs = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %rhs = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<8, !ttcore.tile<32x32, bf16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%lhs, %c0, %c0) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.copy_tile(%rhs, %c0, %c1) : (!ttkernel.cb<8, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// The old operand is the last programmed SrcA CB, not the section's first.
// CHECK-LABEL: func.func @chained_format_switch
// CHECK-DAG: %[[BF16:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[U16:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-DAG: %[[F32:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: ttkernel.reconfig_data_format_srca(%[[BF16]], %[[U16]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[U16]])
// CHECK: ttkernel.reconfig_data_format_srca(%[[U16]], %[[F32]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[F32]])
// CHECK-NOT: ttkernel.reconfig_data_format_srca
// CHECK: ttkernel.tile_regs_release
func.func @chained_format_switch() {
  %bf16 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %u16 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, u16>>
  %f32 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %out = ttkernel.get_compile_time_arg_val(3) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%bf16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.copy_tile(%u16, %c0, %c1) : (!ttkernel.cb<4, !ttcore.tile<32x32, u16>>, index, index) -> ()
  ttkernel.copy_tile(%f32, %c0, %c2) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// transpose_wh_init programs SrcA itself. The following copy reconfigures from
// that operand, not from the common init.
// CHECK-LABEL: func.func @transpose_reprograms_srca
// CHECK-DAG: %[[BF16:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[U16:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-DAG: %[[OUT:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: ttkernel.transpose_wh_init(%[[U16]], %[[OUT]])
// CHECK: ttkernel.reconfig_data_format_srca(%[[U16]], %[[BF16]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[BF16]])
// CHECK-NEXT: ttkernel.copy_tile(%[[BF16]]
// CHECK: ttkernel.tile_regs_release
func.func @transpose_reprograms_srca() {
  %bf16 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %u16 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, u16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%bf16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.transpose_wh_tile(%u16, %c0, %c1) {ttl.transpose_output_cb_index = 2 : index} : (!ttkernel.cb<4, !ttcore.tile<32x32, u16>>, index, index) -> ()
  ttkernel.copy_tile(%bf16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// An existing reconfig already names the programmed operand. Do not emit another.
// CHECK-LABEL: func.func @existing_reconfig
// CHECK: ttkernel.reconfig_data_format_srca
// CHECK-NOT: ttkernel.reconfig_data_format_srca
// CHECK: ttkernel.tile_regs_release
func.func @existing_reconfig() {
  %bf16 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %u16 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, u16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%bf16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.reconfig_data_format_srca(%bf16, %u16) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, u16>>) -> ()
  ttkernel.copy_tile(%u16, %c0, %c1) : (!ttkernel.cb<4, !ttcore.tile<32x32, u16>>, index, index) -> ()
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}
