// Address-based compute uses compile-time formats and one kernel-local configuration context.
// RUN: ttlang-opt %s --convert-ttkernel-to-emitc -o %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp %t.emitc.mlir | FileCheck %s --check-prefix=CPP
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [
  {cb_index = 0 : i64, page_size = 4096 : i64, num_tiles = 1 : i64, block_count = 3 : i64, storage_capacity_pages = 3 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64},
  {cb_index = 1 : i64, page_size = 4096 : i64, num_tiles = 1 : i64, block_count = 3 : i64, storage_capacity_pages = 3 : i64, l1_offset = 8 : i64, l1_payload_offset = 12352 : i64},
  {cb_index = 2 : i64, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 2 : i64, storage_capacity_pages = 2 : i64, l1_offset = 16 : i64, storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>}]},
  {cb_index = 3 : i64, page_size = 512 : i64, num_tiles = 1 : i64, block_count = 2 : i64, storage_capacity_pages = 2 : i64, l1_offset = 24 : i64, l1_payload_offset = 24640 : i64}
]} {
  // SFPU copies preserve the finalized direct-unpack choice in operand metadata.
  // CHECK-LABEL: func.func @compute
  // CHECK: ttlang::l1::target::ComputeContext l1_compute_context;
  // CHECK: get_common_arg_val<uint32_t>(get_compile_time_arg_val(0)) + 0
  // CHECK: get_common_arg_val<uint32_t>(get_compile_time_arg_val(0)) + 8
  // CHECK: ttlang::l1::Operand<static_cast<uint32_t>(DataFormat::Float32), 4096, 32, 32, 1, 3, 3, 64, -1, true>
  // CHECK: l1_compute_context.configure
  // CHECK: ttlang::l1::target::copy_tile
  // CHECK: ttlang::l1::target::pack_tile
  // CHECK-NOT: ttkernel.
  // CHECK-NOT: CircularBuffer
  // CHECK-NOT: cb_wait_front
  // CHECK-NOT: cb_reserve_back
  // CHECK-NOT: cb_push_back
  // CHECK-NOT: cb_pop_front
  // CPP-NOT: CircularBuffer
  // CPP-NOT: cb_wait_front
  // CPP-NOT: cb_reserve_back
  // CPP-NOT: cb_push_back
  // CPP-NOT: cb_pop_front
  // CPP: #ifndef TTLANG_COMPILER_L1_COMPUTE_TARGET_H
  // CPP: ttlang::l1::target::ComputeContext l1_compute_context;
  // CPP: ttlang::l1::Buffer<4096, 1, 3, 3, 64, -1>
  // CPP: ttlang::l1::Operand<static_cast<uint32_t>(DataFormat::Float32), 4096, 32, 32, 1, 3, 3, 64, -1, true>
  // CPP: ttlang::l1::target::copy_tile
  // CPP: ttlang::l1::target::pack_tile
  // CPP-NOT: CircularBuffer
  // CPP-NOT: cb_wait_front
  // CPP-NOT: cb_reserve_back
  // CPP-NOT: cb_push_back
  // CPP-NOT: cb_pop_front
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

  // Sub-tile operands preserve their height and width in the generated C++ type.
  // CHECK-LABEL: func.func @subtile_compute
  // CHECK: ttlang::l1::Operand<static_cast<uint32_t>(DataFormat::Float16_b), 512, 8, 32, 1, 2, 2, 24616, -1, false>
  // CHECK: ttlang::l1::target::copy_tile_init
  // CPP: ttlang::l1::Operand<static_cast<uint32_t>(DataFormat::Float16_b), 512, 8, 32, 1, 2, 2, 24616, -1, false>
  // CPP: ttlang::l1::target::copy_tile_init
  func.func @subtile_compute() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %input = ttkernel.get_compile_time_arg_val(3) : () -> !ttkernel.cb<2, !ttcore.tile<8x32, bf16>>
    ttkernel.copy_tile_init(%input) : (!ttkernel.cb<2, !ttcore.tile<8x32, bf16>>) -> ()
    return
  }

  // Short matmul initialization still requires the invocation context.
  // CHECK-LABEL: func.func @short_compute
  // CHECK: ttlang::l1::target::ComputeContext l1_compute_context;
  // CHECK: l1_compute_context.matmulInitShort
  // CPP: ttlang::l1::target::ComputeContext l1_compute_context;
  // CPP: l1_compute_context.matmulInitShort
  func.func @short_compute() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %lhs = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<3, !ttcore.tile<32x32, f32>>
    %rhs = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<3, !ttcore.tile<32x32, f32>>
    %zero = arith.constant 0 : i32
    ttkernel.mm_init_short(%lhs, %rhs, %zero) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, !ttkernel.cb<3, !ttcore.tile<32x32, f32>>, i32) -> ()
    return
  }

  // Consumer replacement packs to the acquired read window without changing occupancy.
  // CHECK-LABEL: func.func @replace_waited
  // CHECK: ttlang::l1::target::pack_waited_tile
  // CPP: ttlang::l1::target::pack_waited_tile
  func.func @replace_waited() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<3, !ttcore.tile<32x32, f32>>
    %zero = arith.constant 0 : index
    ttkernel.pack_waited_tile(%zero, %storage, %zero, true) {acquired_tiles = 3 : i64} : (index, !ttkernel.cb<3, !ttcore.tile<32x32, f32>>, index) -> ()
    return
  }

  // A tensor-backed DFB can publish its complete capacity as one contiguous transaction.
  // CHECK-LABEL: func.func @publish_tensor_capacity
  // CHECK: ttlang::l1::Buffer<2048, 1, 2, 2, 0, 0>
  // CHECK: .reserve_back({{.*}})
  // CHECK: .push_back({{.*}})
  // CPP: ttlang::l1::Buffer<2048, 1, 2, 2, 0, 0>
  func.func @publish_tensor_capacity() attributes {ttkernel.thread = #ttkernel.thread<noc>, ttl.crta_indices = [0]} {
    %storage = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
    %capacity = arith.constant 2 : i32
    ttkernel.cb_reserve_back(%storage, %capacity) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
    ttkernel.cb_push_back(%storage, %capacity) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
    return
  }
}
