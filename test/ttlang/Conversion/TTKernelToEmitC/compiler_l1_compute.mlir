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
  // Repeated copy transactions retain each acquired window across sequence wrap.
  // CHECK-LABEL: func.func @compute
  // CHECK: ttlang::l1::target::ComputeContext l1_compute_context;
  // CHECK: get_common_arg_val<uint32_t>(get_compile_time_arg_val(0)) + 0
  // CHECK: get_common_arg_val<uint32_t>(get_compile_time_arg_val(0)) + 8
  // CHECK: ttlang::l1::Operand<static_cast<uint32_t>(DataFormat::Float32), 4096, 32, 32, 1, 3, 3, 64, -1, true, true>
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
  // CPP: void push_back
  // CPP: if constexpr (ReloadOwnedSequence)
  // CPP-NEXT: acquiredProducerSequence = loadOwnedSequence(state + published);
  // CPP: void pop_front
  // CPP: if constexpr (ReloadOwnedSequence)
  // CPP-NEXT: acquiredConsumerSequence = loadOwnedSequence(state + consumed);
  // CPP: return address(acquiredProducerSequence);
  // CPP: return address(acquiredConsumerSequence);
  // CPP: #ifndef TTLANG_COMPILER_L1_COMPUTE_TARGET_H
  // CPP: ttlang::l1::target::ComputeContext l1_compute_context;
  // CPP: ttlang::l1::Buffer<4096, 1, 3, 3, 64, -1, false> [[INPUT:cb_ctarg_[0-9]+]]
  // CPP: ttlang::l1::Buffer<4096, 1, 3, 3, 12344, -1, false> [[OUTPUT:cb_ctarg_[0-9]+]]
  // CPP: for (size_t
  // CPP: [[INPUT]].wait_front
  // CPP: [[OUTPUT]].reserve_back
  // CPP: ttlang::l1::target::copy_tile(ttlang::l1::Operand<{{.*}}>([[INPUT]])
  // CPP: ttlang::l1::target::pack_tile<true>({{.*}}ttlang::l1::Operand<{{.*}}>([[OUTPUT]])
  // CPP-NOT: CircularBuffer
  // CPP-NOT: cb_wait_front
  // CPP-NOT: cb_reserve_back
  // CPP-NOT: cb_push_back
  // CPP-NOT: cb_pop_front
  func.func @compute() attributes {ttkernel.thread = #ttkernel.thread<compute>, ttl.unpack_to_dest_fp32 = array<i32: 0>} {
    %input = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<3, !ttcore.tile<32x32, f32>>
    %output = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<3, !ttcore.tile<32x32, f32>>
    %zero = arith.constant 0 : index
    %one_index = arith.constant 1 : index
    %seven = arith.constant 7 : index
    %one_i32 = arith.constant 1 : i32
    ttkernel.unary_op_init_common(%input, %output) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, !ttkernel.cb<3, !ttcore.tile<32x32, f32>>) -> ()
    ttkernel.copy_tile_init(%input) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>) -> ()
    scf.for %iteration = %zero to %seven step %one_index {
      ttkernel.cb_wait_front(%input, %one_i32) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, i32) -> ()
      ttkernel.cb_reserve_back(%output, %one_i32) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, i32) -> ()
      ttkernel.tile_regs_acquire() : () -> ()
      ttkernel.copy_tile(%input, %zero, %zero) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, index, index) -> ()
      ttkernel.tile_regs_commit() : () -> ()
      ttkernel.tile_regs_wait() : () -> ()
      ttkernel.pack_tile(%zero, %output, %zero, true) : (index, !ttkernel.cb<3, !ttcore.tile<32x32, f32>>, index) -> ()
      ttkernel.tile_regs_release() : () -> ()
      ttkernel.cb_push_back(%output, %one_i32) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, i32) -> ()
      ttkernel.cb_pop_front(%input, %one_i32) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, i32) -> ()
    }
    return
  }

  // Address-bearing operations use the DFB object that captured the acquired sequence.
  // CHECK-LABEL: func.func @address_operations
  // CHECK: emitc.call_opaque "ttlang::l1::target::add_tiles"
  // CHECK-SAME: ttlang.requires_compiler_l1
  // CPP: void kernel_main()
  // CPP: ttlang::l1::Buffer<4096, 1, 3, 3, 64, -1, false> [[ADDRESS_INPUT:cb_ctarg_[0-9]+]]
  // CPP: [[ADDRESS_INPUT]].wait_front
  // CPP: ttlang::l1::target::add_tiles(ttlang::l1::Operand<{{.*}}>([[ADDRESS_INPUT]]), ttlang::l1::Operand<{{.*}}>([[ADDRESS_INPUT]])
  // CPP: ttlang::l1::target::sub_tiles(ttlang::l1::Operand<{{.*}}>([[ADDRESS_INPUT]]), ttlang::l1::Operand<{{.*}}>([[ADDRESS_INPUT]])
  // CPP: ttlang::l1::target::mul_tiles(ttlang::l1::Operand<{{.*}}>([[ADDRESS_INPUT]]), ttlang::l1::Operand<{{.*}}>([[ADDRESS_INPUT]])
  // CPP: ttlang::l1::target::binary_dest_reuse_tiles<{{.*}}>(ttlang::l1::Operand<{{.*}}>([[ADDRESS_INPUT]])
  // CPP: ttlang::l1::target::unary_bcast<{{.*}}>(ttlang::l1::Operand<{{.*}}>([[ADDRESS_INPUT]])
  // CPP: ttlang::l1::target::reduce_tile<{{.*}}>(ttlang::l1::Operand<{{.*}}>([[ADDRESS_INPUT]]), ttlang::l1::Operand<{{.*}}>([[ADDRESS_INPUT]])
  // CPP: ttlang::l1::target::matmul_tiles(ttlang::l1::Operand<{{.*}}>([[ADDRESS_INPUT]]), ttlang::l1::Operand<{{.*}}>([[ADDRESS_INPUT]])
  // CPP: ttlang::l1::target::matmul_block(ttlang::l1::Operand<{{.*}}>([[ADDRESS_INPUT]]), ttlang::l1::Operand<{{.*}}>([[ADDRESS_INPUT]])
  // CPP: ttlang::l1::target::matmul_block_strided(ttlang::l1::Operand<{{.*}}>([[ADDRESS_INPUT]]), ttlang::l1::Operand<{{.*}}>([[ADDRESS_INPUT]])
  // CPP: ttlang::l1::target::transpose_wh_tile(ttlang::l1::Operand<{{.*}}>([[ADDRESS_INPUT]])
  func.func @address_operations() attributes {ttkernel.thread = #ttkernel.thread<compute>, ttl.unpack_to_dest_fp32 = array<i32: 0>} {
    %input = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<3, !ttcore.tile<32x32, f32>>
    %zero = arith.constant 0 : index
    %zero_i32 = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    ttkernel.cb_wait_front(%input, %one) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, i32) -> ()
    ttkernel.add_tiles(%input, %input, %zero, %zero, %zero) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, !ttkernel.cb<3, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
    ttkernel.sub_tiles(%input, %input, %zero, %zero, %zero) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, !ttkernel.cb<3, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
    ttkernel.mul_tiles(%input, %input, %zero, %zero, %zero) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, !ttkernel.cb<3, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
    ttkernel.binary_dest_reuse_tiles(%input, %zero, %zero, <add>, <dest_to_srca>) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, index, index) -> ()
    ttkernel.unary_bcast(%input, %zero, %zero, <scalar>) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, index, index) -> ()
    ttkernel.reduce_tile(%input, %input, %zero, %zero, %zero, <reduce_sum>, <reduce_dim_scalar>) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, !ttkernel.cb<3, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
    ttkernel.matmul_tiles(%input, %input, %zero, %zero, %zero) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, !ttkernel.cb<3, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
    ttkernel.matmul_block(%input, %input, %zero, %zero, %zero, %zero_i32, %one, %one, %one) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, !ttkernel.cb<3, !ttcore.tile<32x32, f32>>, index, index, index, i32, i32, i32, i32) -> ()
    "ttkernel.experimental.matmul_block"(%input, %input, %zero, %zero, %zero, %zero_i32, %one, %one, %one, %one) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, !ttkernel.cb<3, !ttcore.tile<32x32, f32>>, index, index, index, i32, i32, i32, i32, i32) -> ()
    ttkernel.transpose_wh_tile(%input, %zero, %zero) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, index, index) -> ()
    return
  }

  // Sub-tile operands preserve their height and width in the generated C++ type.
  // CHECK-LABEL: func.func @subtile_compute
  // CHECK: ttlang::l1::Operand<static_cast<uint32_t>(DataFormat::Float16_b), 512, 8, 32, 1, 2, 2, 24616, -1, false, true>
  // CHECK: ttlang::l1::target::copy_tile_init
  // CPP: ttlang::l1::Operand<static_cast<uint32_t>(DataFormat::Float16_b), 512, 8, 32, 1, 2, 2, 24616, -1, false, true>
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
  // CPP: ttlang::l1::Buffer<4096, 1, 3, 3, 64, -1, false> [[WAITED:cb_ctarg_[0-9]+]]
  // CPP: [[WAITED]].wait_front
  // CPP: ttlang::l1::target::pack_waited_tile<true>({{.*}}ttlang::l1::Operand<{{.*}}>([[WAITED]])
  func.func @replace_waited() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<3, !ttcore.tile<32x32, f32>>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : i32
    ttkernel.cb_wait_front(%storage, %one) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, i32) -> ()
    ttkernel.pack_waited_tile(%zero, %storage, %zero, true) {acquired_tiles = 3 : i64} : (index, !ttkernel.cb<3, !ttcore.tile<32x32, f32>>, index) -> ()
    return
  }

  // A tensor-backed DFB can publish its complete capacity as one contiguous transaction.
  // CHECK-LABEL: func.func @publish_tensor_capacity
  // CHECK: ttlang::l1::Buffer<2048, 1, 2, 2, 0, 0, false>
  // CHECK: .reserve_back({{.*}})
  // CHECK: .push_back({{.*}})
  // CPP: ttlang::l1::Buffer<2048, 1, 2, 2, 0, 0, false>
  func.func @publish_tensor_capacity() attributes {ttkernel.thread = #ttkernel.thread<noc>, ttl.crta_indices = [0]} {
    %storage = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
    %capacity = arith.constant 2 : i32
    ttkernel.cb_reserve_back(%storage, %capacity) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
    ttkernel.cb_push_back(%storage, %capacity) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
    return
  }

  // Proven transfer completion suppresses only the redundant release barrier.
  // CHECK-LABEL: func.func @completed_transfer_release
  // CHECK: .push_back<true>({{.*}})
  // CHECK: .pop_front<true>({{.*}})
  // CPP: .push_back<true>({{.*}})
  // CPP: .pop_front<true>({{.*}})
  func.func @completed_transfer_release() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<3, !ttcore.tile<32x32, f32>>
    %one = arith.constant 1 : i32
    ttkernel.cb_push_back(%storage, %one) {payload_complete}
        : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, i32) -> ()
    ttkernel.cb_pop_front(%storage, %one) {payload_complete}
        : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, i32) -> ()
    return
  }

  // Opaque DFB users retain owned-sequence reloads because they may advance the same interface.
  // CHECK-LABEL: func.func @opaque_dfb_user
  // CHECK: ttlang::l1::Buffer<4096, 1, 3, 3, 64, -1, true>
  // CPP: ttlang::l1::Buffer<4096, 1, 3, 3, 64, -1, true>
  func.func @opaque_dfb_user() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<3, !ttcore.tile<32x32, f32>>
    %one = arith.constant 1 : i32
    ttkernel.opaque_call "advance"() {dfb_resource_indices = array<i32: 0>, header = "advance.hpp"} : () -> ()
    ttkernel.cb_reserve_back(%storage, %one) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, i32) -> ()
    return
  }

  // An opaque user invalidates only the DFB indices declared by its effect contract.
  // CHECK-LABEL: func.func @opaque_other_dfb_user
  // CHECK: ttlang::l1::Buffer<4096, 1, 3, 3, 64, -1, false>
  // CHECK: ttlang::l1::Buffer<4096, 1, 3, 3, 12344, -1, true>
  func.func @opaque_other_dfb_user() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %unaffected = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<3, !ttcore.tile<32x32, f32>>
    %affected = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<3, !ttcore.tile<32x32, f32>>
    %one = arith.constant 1 : i32
    ttkernel.opaque_call "advance"() {dfb_resource_indices = array<i32: 1>, header = "advance.hpp"} : () -> ()
    ttkernel.cb_reserve_back(%unaffected, %one) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, i32) -> ()
    ttkernel.cb_reserve_back(%affected, %one) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, i32) -> ()
    return
  }

  // Function calls retain owned-sequence reloads because callees may advance a DFB interface.
  // CHECK-LABEL: func.func @function_dfb_user
  // CHECK: ttlang::l1::Buffer<4096, 1, 3, 3, 64, -1, true>
  // CPP: ttlang::l1::Buffer<4096, 1, 3, 3, 64, -1, true>
  func.func @function_dfb_user() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<3, !ttcore.tile<32x32, f32>>
    %one = arith.constant 1 : i32
    func.call @unknown() : () -> ()
    ttkernel.cb_reserve_back(%storage, %one) : (!ttkernel.cb<3, !ttcore.tile<32x32, f32>>, i32) -> ()
    return
  }

  func.func private @unknown()

  // Frontend Boolean negation is a pure expression and does not obscure DFB effects.
  // CHECK-LABEL: func.func @logical_not
  // CHECK: emitc.logical_not
  func.func @logical_not() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %condition = arith.constant true
    %negated = emitc.logical_not %condition : i1
    return
  }
}
