// copy_tile_init does not program the SrcA data format. The common init does,
// once, and the format persists across sync regions. A later copy from a
// different element type must reconfigure SrcA or the unpacker keeps the
// programmed format. Tests cover straight-line switches, full configures that
// reprogram SrcA, short inits that require it, loops, branches, hoisted inits,
// and regions that share one hoisted common init.
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

// An existing reconfig changes SrcA between two copies from one CB. The SrcA
// write invalidates the copy init, so the second copy gets its own init and
// the format is restored before it.
// CHECK-LABEL: func.func @reconfig_between_same_copies
// CHECK-DAG: %[[BF16:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[U16:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.copy_tile_init(%[[BF16]])
// CHECK-NEXT: ttkernel.copy_tile(%[[BF16]]
// CHECK-NEXT: ttkernel.reconfig_data_format_srca(%[[U16]]) :
// CHECK-NEXT: ttkernel.reconfig_data_format_srca(%[[U16]], %[[BF16]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[BF16]])
// CHECK-NEXT: ttkernel.copy_tile(%[[BF16]]
// CHECK: ttkernel.tile_regs_release
func.func @reconfig_between_same_copies() {
  %bf16 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %u16 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, u16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%bf16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.reconfig_data_format_srca(%u16) : (!ttkernel.cb<4, !ttcore.tile<32x32, u16>>) -> ()
  ttkernel.copy_tile(%bf16, %c0, %c1) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// A SrcA write forgets the init key but not the pending reduce_uninit.
// CHECK-LABEL: func.func @reconfig_keeps_reduce_uninit
// CHECK: ttkernel.reduce_tile
// CHECK-NEXT: ttkernel.reconfig_data_format_srca
// CHECK-NEXT: ttkernel.reduce_uninit
// CHECK-NEXT: ttkernel.tile_regs_commit
func.func @reconfig_keeps_reduce_uninit() {
  %bf16 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %scaler = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.reduce_tile(%bf16, %scaler, %c0, %c0, %c0, <reduce_sum>, <reduce_dim_col>) {ttl.reduce_output_cb_index = 2 : index} : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
  ttkernel.reconfig_data_format_srca(%bf16) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// Full matmul init programs SrcA from in1 (SrcOrder::Reverse), not in0.
// CHECK-LABEL: func.func @matmul_srca_is_in1
// CHECK-DAG: %[[IN0:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[IN1:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.mm_block_init"(%[[IN0]], %[[IN1]]
// CHECK: ttkernel.reconfig_data_format_srca(%[[IN1]], %[[IN0]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[IN0]])
// CHECK-NEXT: ttkernel.copy_tile(%[[IN0]]
// CHECK: ttkernel.tile_regs_release
func.func @matmul_srca_is_in1() {
  %in0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %in1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, u16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %i0 = arith.constant 0 : i32
  %i1 = arith.constant 1 : i32
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.matmul_block(%in0, %in1, %c0, %c0, %c0, %i0, %i1, %i1, %i1) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, u16>>, index, index, index, i32, i32, i32, i32) -> ()
  ttkernel.copy_tile(%in0, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// Block copy uses copy_tile_init. The reconfig precedes that init.
// CHECK-LABEL: func.func @block_copy_reconfig_before_init
// CHECK-DAG: %[[BF16:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[U16:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.reconfig_data_format_srca(%[[BF16]], %[[U16]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[U16]])
// CHECK-NEXT: ttkernel.copy_block_matmul_partials(%[[U16]]
// CHECK: ttkernel.tile_regs_release
func.func @block_copy_reconfig_before_init() {
  %bf16 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %u16 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, u16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%bf16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.copy_block_matmul_partials(%u16, %c0, %c1, %c4) : (!ttkernel.cb<4, !ttcore.tile<32x32, u16>>, index, index, index) -> ()
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// The backedge leaves U16, so the next iteration's BF16 copy cannot reuse the
// preheader operand. The one-operand form always reconfigures. Each copy also
// gets its own copy_tile_init: the reconfig changes the format and not the
// copy unpack mode.
// CHECK-LABEL: func.func @loop_backedge_reconfig
// CHECK-DAG: %[[BF16:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[U16:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: scf.for
// CHECK-NEXT: ttkernel.reconfig_data_format_srca(%[[BF16]]) :
// CHECK-NEXT: ttkernel.copy_tile_init(%[[BF16]])
// CHECK-NEXT: ttkernel.copy_tile(%[[BF16]]
// CHECK-NEXT: ttkernel.reconfig_data_format_srca(%[[BF16]], %[[U16]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[U16]])
// CHECK-NEXT: ttkernel.copy_tile(%[[U16]]
// CHECK: ttkernel.tile_regs_release
func.func @loop_backedge_reconfig() {
  %bf16 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %u16 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, u16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  ttkernel.tile_regs_acquire() : () -> ()
  scf.for %iv = %c0 to %c4 step %c1 {
    ttkernel.copy_tile(%bf16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    ttkernel.copy_tile(%u16, %c0, %c1) : (!ttkernel.cb<4, !ttcore.tile<32x32, u16>>, index, index) -> ()
  }
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// Branch exits disagree, so the copy after the join uses the one-operand form.
// Each branch copy gets the init beside it.
// CHECK-LABEL: func.func @branch_join_reconfig
// CHECK-DAG: %[[BF16:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[U16:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-DAG: %[[F32:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: scf.if
// CHECK-NEXT: ttkernel.copy_tile_init(%[[U16]])
// CHECK-NEXT: ttkernel.copy_tile(%[[U16]]
// CHECK: ttkernel.reconfig_data_format_srca(%[[U16]], %[[F32]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[F32]])
// CHECK-NEXT: ttkernel.copy_tile(%[[F32]]
// CHECK: ttkernel.reconfig_data_format_srca(%[[BF16]]) :
// CHECK-NEXT: ttkernel.copy_tile_init(%[[BF16]])
// CHECK-NEXT: ttkernel.copy_tile(%[[BF16]]
// CHECK: ttkernel.tile_regs_release
func.func @branch_join_reconfig() {
  %bf16 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %u16 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, u16>>
  %f32 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %out = ttkernel.get_compile_time_arg_val(3) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cond = arith.constant true
  ttkernel.tile_regs_acquire() : () -> ()
  scf.if %cond {
    ttkernel.copy_tile(%u16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, u16>>, index, index) -> ()
    scf.yield
  } else {
    ttkernel.copy_tile(%f32, %c0, %c1) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
    scf.yield
  }
  ttkernel.copy_tile(%bf16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// add_tiles_init asserts SrcA already matches in0. The copy switches SrcA to
// BF16, so the add reconfigures back to U16 before its init, and the final
// copy reconfigures again.
// CHECK-LABEL: func.func @add_init_leaves_srca
// CHECK-DAG: %[[BF16:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[U16:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.reconfig_data_format_srca(%[[U16]], %[[BF16]])
// CHECK: ttkernel.reconfig_data_format_srca(%[[BF16]], %[[U16]])
// CHECK-NEXT: ttkernel.add_tiles_init(%[[U16]]
// CHECK: ttkernel.reconfig_data_format_srca(%[[U16]], %[[BF16]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[BF16]])
// CHECK: ttkernel.tile_regs_release
func.func @add_init_leaves_srca() {
  %bf16 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %u16 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, u16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%bf16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.add_tiles(%u16, %u16, %c0, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, u16>>, !ttkernel.cb<4, !ttcore.tile<32x32, u16>>, index, index, index) -> ()
  ttkernel.copy_tile(%bf16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// Row sum unpacks the scaler through SrcA. reduce_init does not write that
// format, so the scaler is reconfigured before the init. The following copy
// reconfigures back.
// CHECK-LABEL: func.func @reduce_init_leaves_srca
// CHECK-DAG: %[[BF16:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[U16:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.reconfig_data_format_srca(%[[BF16]], %[[U16]])
// CHECK-NEXT: ttkernel.reduce_init
// CHECK: ttkernel.reconfig_data_format_srca(%[[U16]], %[[BF16]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[BF16]])
// CHECK: ttkernel.tile_regs_release
func.func @reduce_init_leaves_srca() {
  %bf16 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %scaler = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, u16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%bf16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.reduce_tile(%bf16, %scaler, %c0, %c0, %c0, <reduce_sum>, <reduce_dim_row>) {ttl.reduce_output_cb_index = 2 : index} : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, u16>>, index, index, index) -> ()
  ttkernel.copy_tile(%bf16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// A copy leaves SrcA on in0. mm_block_init_short does not restore in1, so
// the matmul reconfigures before that init.
// CHECK-LABEL: func.func @matmul_restores_srca
// CHECK-DAG: %[[IN0:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[IN1:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.reconfig_data_format_srca(%[[IN1]], %[[IN0]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[IN0]])
// CHECK: ttkernel.reconfig_data_format_srca(%[[IN0]], %[[IN1]])
// CHECK-NEXT: "ttkernel.mm_block_init_short"(%[[IN0]], %[[IN1]]
// CHECK: ttkernel.tile_regs_release
func.func @matmul_restores_srca() {
  %in0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %in1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, u16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %i0 = arith.constant 0 : i32
  %i1 = arith.constant 1 : i32
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%in0, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.matmul_block(%in0, %in1, %c0, %c0, %c0, %i0, %i1, %i1, %i1) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, u16>>, index, index, index, i32, i32, i32, i32) -> ()
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// Both common inits hoist above the compiler loop, so the U16 one is what the
// hardware holds when the loop starts, and the backedge carries U16 as well.
// The first region reconfigures to BF16 and the second back to U16.
// CHECK-LABEL: func.func @two_regions_in_compiler_loop
// CHECK-DAG: %[[BF16:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[U16:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-DAG: %[[OUT:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: ttkernel.init_sfpu(%[[BF16]], %[[OUT]])
// CHECK-NEXT: ttkernel.init_sfpu(%[[U16]], %[[OUT]])
// CHECK-NEXT: scf.for
// CHECK-NEXT: ttkernel.tile_regs_acquire
// CHECK-NEXT: ttkernel.reconfig_data_format_srca(%[[U16]], %[[BF16]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[BF16]])
// CHECK-NEXT: ttkernel.copy_tile(%[[BF16]]
// CHECK: ttkernel.tile_regs_acquire
// CHECK-NEXT: ttkernel.reconfig_data_format_srca(%[[BF16]], %[[U16]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[U16]])
// CHECK-NEXT: ttkernel.copy_tile(%[[U16]]
// CHECK: ttkernel.tile_regs_release
func.func @two_regions_in_compiler_loop() {
  %bf16 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %u16 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, u16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %iv = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.copy_tile(%bf16, %iv, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.copy_tile(%u16, %iv, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, u16>>, index, index) -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.tile_loop_stride = 1 : index}
  func.return
}

// No pack means no common init, so nothing has programmed SrcA. The first copy
// uses the one-operand form; the second knows the programmed operand.
// CHECK-LABEL: func.func @no_common_init
// CHECK-DAG: %[[BF16:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[U16:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-NOT: ttkernel.init_sfpu
// CHECK: ttkernel.reconfig_data_format_srca(%[[BF16]]) :
// CHECK-NEXT: ttkernel.copy_tile_init(%[[BF16]])
// CHECK-NEXT: ttkernel.copy_tile(%[[BF16]]
// CHECK-NEXT: ttkernel.reconfig_data_format_srca(%[[BF16]], %[[U16]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[U16]])
// CHECK-NEXT: ttkernel.copy_tile(%[[U16]]
// CHECK: ttkernel.tile_regs_release
func.func @no_common_init() {
  %bf16 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %u16 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, u16>>
  %c0 = arith.constant 0 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%bf16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.copy_tile(%u16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, u16>>, index, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// Nested loops that keep one copy key hoist copy_tile_init above the outer
// loop. The reconfig is placed in front of that hoisted init, once, and uses
// the operand programmed at that point.
// CHECK-LABEL: func.func @hoisted_copy_init_nested_loops
// CHECK-DAG: %[[BF16:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[U16:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.copy_tile(%[[BF16]]
// CHECK-NEXT: ttkernel.reconfig_data_format_srca(%[[BF16]], %[[U16]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[U16]])
// CHECK-NEXT: scf.for
// CHECK-NEXT: scf.for
// CHECK-NEXT: ttkernel.copy_tile(%[[U16]]
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NOT: ttkernel.reconfig_data_format_srca
// CHECK: ttkernel.tile_regs_release
func.func @hoisted_copy_init_nested_loops() {
  %bf16 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %u16 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, u16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%bf16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      ttkernel.copy_tile(%u16, %j, %j) : (!ttkernel.cb<4, !ttcore.tile<32x32, u16>>, index, index) -> ()
    }
  }
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// A loop-local op between the outer and inner loop does not separate the
// hoisted init from its reconfig: both precede the outer loop.
// CHECK-LABEL: func.func @hoisted_copy_init_loop_local_op
// CHECK-DAG: %[[BF16:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[U16:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.copy_tile(%[[BF16]]
// CHECK-NEXT: ttkernel.reconfig_data_format_srca(%[[BF16]], %[[U16]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[U16]])
// CHECK-NEXT: scf.for
// CHECK-NEXT: arith.addi
// CHECK-NEXT: scf.for
// CHECK-NEXT: ttkernel.copy_tile(%[[U16]]
// CHECK-NOT: ttkernel.reconfig_data_format_srca
// CHECK: ttkernel.tile_regs_release
func.func @hoisted_copy_init_loop_local_op() {
  %bf16 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %u16 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, u16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%bf16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  scf.for %i = %c0 to %c4 step %c1 {
    %row = arith.addi %i, %c1 : index
    scf.for %j = %c0 to %c4 step %c1 {
      ttkernel.copy_tile(%u16, %row, %j) : (!ttkernel.cb<4, !ttcore.tile<32x32, u16>>, index, index) -> ()
    }
  }
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// The while body may run zero or more times, so both its entry and the exit
// after the loop have no single programmed operand.
// CHECK-LABEL: func.func @while_loop_reconfig
// CHECK-DAG: %[[BF16:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[U16:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: } do {
// CHECK-NEXT: ttkernel.reconfig_data_format_srca(%[[U16]]) :
// CHECK-NEXT: ttkernel.copy_tile_init(%[[U16]])
// CHECK-NEXT: ttkernel.copy_tile(%[[U16]]
// CHECK: ttkernel.reconfig_data_format_srca(%[[BF16]]) :
// CHECK-NEXT: ttkernel.copy_tile_init(%[[BF16]])
// CHECK-NEXT: ttkernel.copy_tile(%[[BF16]]
// CHECK: ttkernel.tile_regs_release
func.func @while_loop_reconfig(%cond: i1) {
  %bf16 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %u16 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, u16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%bf16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  scf.while : () -> () {
    scf.condition(%cond)
  } do {
    ttkernel.copy_tile(%u16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, u16>>, index, index) -> ()
    scf.yield
  }
  ttkernel.copy_tile(%bf16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// One branch ends in reduce and the other does not. The join still has to
// clear the packer edge mask before commit.
// CHECK-LABEL: func.func @reduce_uninit_at_branch_join
// CHECK: } else {
// CHECK: ttkernel.copy_tile
// CHECK: ttkernel.reduce_uninit
// CHECK-NEXT: ttkernel.tile_regs_commit
func.func @reduce_uninit_at_branch_join() {
  %bf16 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %scaler = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %cond = arith.constant true
  ttkernel.tile_regs_acquire() : () -> ()
  scf.if %cond {
    ttkernel.reduce_tile(%bf16, %scaler, %c0, %c0, %c0, <reduce_sum>, <reduce_dim_col>) {ttl.reduce_output_cb_index = 2 : index} : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
    scf.yield
  } else {
    ttkernel.copy_tile(%bf16, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    scf.yield
  }
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// The loop may not run, so a reduce inside it still requires reduce_uninit
// before the following commit.
// CHECK-LABEL: func.func @reduce_uninit_when_loop_may_skip
// CHECK: ttkernel.reduce_init
// CHECK-NEXT: scf.for
// CHECK: ttkernel.reduce_tile
// CHECK: ttkernel.reduce_uninit
// CHECK-NEXT: ttkernel.tile_regs_commit
func.func @reduce_uninit_when_loop_may_skip(%n: index) {
  %bf16 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %scaler = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ttkernel.tile_regs_acquire() : () -> ()
  scf.for %i = %c0 to %n step %c1 {
    ttkernel.reduce_tile(%bf16, %scaler, %c0, %c0, %c0, <reduce_sum>, <reduce_dim_col>) {ttl.reduce_output_cb_index = 2 : index} : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
  }
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}
