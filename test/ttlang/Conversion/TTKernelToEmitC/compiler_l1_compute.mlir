// Address-based compute uses compile-time formats and one invocation context.
// RUN: ttlang-opt %s --convert-ttkernel-to-emitc | FileCheck %s
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [
  {cb_index = 0 : i64, page_size = 4096 : i64, num_tiles = 1 : i64, block_count = 3 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64},
  {cb_index = 1 : i64, page_size = 4096 : i64, num_tiles = 1 : i64, block_count = 3 : i64, l1_offset = 8 : i64, l1_payload_offset = 12352 : i64}
]} {
  // SFPU copies preserve the finalized direct-unpack choice in operand metadata.
  // CHECK-LABEL: func.func @compute
  // CHECK: ttlang::l1::target::ComputeContext l1_compute_context;
  // CHECK: get_common_arg_val<uint32_t>(get_compile_time_arg_val(0)) + 0
  // CHECK: get_common_arg_val<uint32_t>(get_compile_time_arg_val(0)) + 8
  // CHECK: ttlang::l1::Operand<static_cast<uint32_t>(DataFormat::Float32), 4096, 1, 3, 64, true>
  // CHECK: l1_compute_context.configure
  // CHECK: ttlang::l1::target::copy_tile
  // CHECK: ttlang::l1::target::pack_tile
  // CHECK-NOT: ttkernel.
  func.func @compute() attributes {ttkernel.thread = #ttkernel.thread<compute>, ttl.unpack_to_dest_fp32 = array<i32: 0>} {
    %input = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<3, !ttcore.tile<32x32, f32>>
    %output = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<3, !ttcore.tile<32x32, f32>>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : i32
    ttkernel.cb_wait_front(%input, %one) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, i32) -> ()
    ttkernel.cb_reserve_back(%output, %one) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, i32) -> ()
    ttkernel.unary_op_init_common(%input, %output) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, !ttkernel.cb<3, !ttcore.tile<32x32, f32>>) -> ()
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.copy_tile_init(%input) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>) -> ()
    ttkernel.copy_tile(%input, %zero, %zero) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, index, index) -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%zero, %output, %zero, true) : (index, !ttkernel.cb<3, !ttcore.tile<32x32, f32>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
    ttkernel.cb_push_back(%output, %one) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, i32) -> ()
    ttkernel.cb_pop_front(%input, %one) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, i32) -> ()
    return
  }
}
