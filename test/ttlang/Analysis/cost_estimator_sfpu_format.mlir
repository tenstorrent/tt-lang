// RUN: ttlang-opt --split-input-file --pass-pipeline='builtin.module(ttkernel-cost-estimate{detail=1})' %s -o /dev/null 2>&1 | FileCheck %s

// The format an SFPU operation was measured on, which it does not name itself.
//
// `exp_tile` takes a DST index and no circular buffer: the tile it works on
// reached DST several operations earlier. The measurement is still keyed on the
// formats the engines saw, so the format comes from the kernel's own buffers --
// the ones the packer writes are the output side, the rest the input side.

// A kernel that reads bf16 and packs f32. The two formats are not a conflict,
// they are the pair the row was measured at, so the SFPU operations resolve.
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @unary_bf16_to_f32() attributes {
      dst_full_sync_en = false,
      fp32_dest_acc_en = false,
      ttkernel.thread = #ttkernel.thread<compute>} {
    %c0 = arith.constant 0 : index
    %in = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
    %out = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.copy_tile_init(%in) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%in, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    ttkernel.exp_tile_init() {approx = true} : () -> ()
    ttkernel.exp_tile(%c0) {approx = true, iterations = 8 : i32} : (index) -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %out, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
    return
  }
}

// CHECK: cost estimate: 7 of 11 placements measured
// CHECK-NEXT: 0 unmatched {{.*}}, 4 untimed
// CHECK: TRISC1 math
// CHECK: ttkernel.copy_tile {{.*}} 19 {{.*}} meas
// CHECK: ttkernel.exp_tile_init {{.*}} 88 {{.*}} meas
// CHECK-NEXT: ttkernel.exp_tile {{.*}} 112 {{.*}} meas

// -----

// The same kernel reading two formats. Which one the exponential saw is no
// longer decidable from the buffers, so it answers nothing rather than guessing
// -- while `add_tiles`, which names its own buffer, still resolves.
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @binary_mixed_inputs() attributes {
      dst_full_sync_en = false,
      fp32_dest_acc_en = false,
      ttkernel.thread = #ttkernel.thread<compute>} {
    %c0 = arith.constant 0 : index
    %lhs = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
    %rhs = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
    %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.binary_op_init_common(%lhs, %rhs, %out) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>) -> ()
    ttkernel.add_tiles_init(%lhs, %rhs) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>) -> ()
    ttkernel.add_tiles(%lhs, %rhs, %c0, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
    ttkernel.exp_tile_init() {approx = true} : () -> ()
    ttkernel.exp_tile(%c0) {approx = true, iterations = 8 : i32} : (index) -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %out, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
    return
  }
}

// CHECK: cost estimate:
// CHECK: TRISC1 math
// CHECK: ttkernel.add_tiles {{.*}} meas
// CHECK: ttkernel.exp_tile_init {{.*}} nokey
// CHECK-NEXT: ttkernel.exp_tile {{.*}} nokey
