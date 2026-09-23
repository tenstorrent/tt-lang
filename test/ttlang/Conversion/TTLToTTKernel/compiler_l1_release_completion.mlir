// Verifies that compiler-managed SRAM releases record proven payload completion.
// RUN: ttlang-opt %s --split-input-file --convert-ttl-to-ttkernel | FileCheck %s

#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>

module attributes {ttl.dfb_allocations = [{block_count = 2 : i32,
  dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, f32>,
  l1_allocation_bytes = 8192 : i64, l1_offset = 0 : i64,
  l1_payload_offset = 64 : i64, num_tiles = 1 : i32,
  page_size = 4096 : i32, storage_index = 0 : i32}],
  ttl.l1_arena_bytes = 8256 : i64, ttl.launch_grid = [1, 1],
  ttl.memory_model = "compiler-l1", ttl.target_arch = #ttcore.arch<blackhole>} {

// A waited NoC read completes the only write to each reserved transaction.
// CHECK-LABEL: func.func @completed_producer_transactions
// CHECK: scf.for
// CHECK: ttkernel.noc_async_read_barrier
// CHECK-NEXT: ttkernel.cb_push_back{{.*}}payload_complete
func.func @completed_producer_transactions(
    %source: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>)
    attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0],
                ttl.kernel_thread = #ttkernel.thread<noc>} {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %storage = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  scf.for %iteration = %zero to %two step %one {
    %reserved = ttl.cb_reserve %storage
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %slice = ttl.tensor_slice %source[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
          -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %transfer = ttl.copy %slice, %storage
        : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
          -> !ttl.transfer_handle<read>
    ttl.wait %transfer : !ttl.transfer_handle<read>
    ttl.cb_push %storage : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  }
  return
}

// A waited NoC write completes the only read from each consumed transaction.
// CHECK-LABEL: func.func @completed_consumer_transactions
// CHECK: scf.for
// CHECK: ttkernel.noc_async_write_barrier
// CHECK-NEXT: ttkernel.cb_pop_front{{.*}}payload_complete
func.func @completed_consumer_transactions(
    %destination: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>)
    attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0],
                ttl.kernel_thread = #ttkernel.thread<noc>} {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %storage = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  scf.for %iteration = %zero to %two step %one {
    %waited = ttl.cb_wait %storage
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %slice = ttl.tensor_slice %destination[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
          -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %transfer = ttl.copy %storage, %slice
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           tensor<1x1x!ttcore.tile<32x32, f32>, #layout>)
          -> !ttl.transfer_handle<write>
    ttl.wait %transfer : !ttl.transfer_handle<write>
    ttl.cb_pop %storage : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  }
  return
}

// An additional payload access after the wait requires release-time completion.
// CHECK-LABEL: func.func @uncompleted_producer_transaction
// CHECK: ttkernel.noc_async_read_barrier
// CHECK: ttkernel.noc_async_read_tile
// CHECK: ttkernel.cb_push_back(%{{.*}}, %{{.*}}) :
// CHECK: return
func.func @uncompleted_producer_transaction(
    %source: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>)
    attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0],
                ttl.kernel_thread = #ttkernel.thread<noc>} {
  %zero = arith.constant 0 : index
  %storage = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %reserved = ttl.cb_reserve %storage
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %slice = ttl.tensor_slice %source[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
        -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
  %completed = ttl.copy %slice, %storage
      : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> !ttl.transfer_handle<read>
  ttl.wait %completed : !ttl.transfer_handle<read>
  %pending = ttl.copy %slice, %storage
      : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> !ttl.transfer_handle<read>
  ttl.cb_push %storage : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  return
}

// A partial release cannot use completion proved for a larger acquisition.
// CHECK-LABEL: func.func @partial_producer_release
// CHECK: ttkernel.noc_async_read_barrier
// CHECK-NEXT: ttkernel.cb_push_back(%{{.*}}, %{{.*}}) :
// CHECK: return
func.func @partial_producer_release(
    %source: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>)
    attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0],
                ttl.kernel_thread = #ttkernel.thread<noc>} {
  %zero = arith.constant 0 : index
  %storage = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %reserved = ttl.cb_reserve %storage {num_tiles = 2 : i64}
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<2x1x1x!ttcore.tile<32x32, f32>>
  %slice = ttl.tensor_slice %source[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
        -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
  %transfer = ttl.copy %slice, %storage
      : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> !ttl.transfer_handle<read>
  ttl.wait %transfer : !ttl.transfer_handle<read>
  ttl.cb_push %storage {num_tiles = 1 : i64}
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  return
}

// A control-flow-selected handle remains conservative even when both values
// currently name the same copy.
// CHECK-LABEL: func.func @indirect_completion_handle
// CHECK: ttkernel.noc_async_read_barrier
// CHECK-NEXT: ttkernel.cb_push_back(%{{.*}}, %{{.*}}) :
// CHECK: return
func.func @indirect_completion_handle(
    %source: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, %condition: i1)
    attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0],
                ttl.kernel_thread = #ttkernel.thread<noc>} {
  %zero = arith.constant 0 : index
  %storage = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %reserved = ttl.cb_reserve %storage
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %slice = ttl.tensor_slice %source[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
        -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
  %transfer = ttl.copy %slice, %storage
      : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> !ttl.transfer_handle<read>
  %selected = scf.if %condition -> !ttl.transfer_handle<read> {
    scf.yield %transfer : !ttl.transfer_handle<read>
  } else {
    scf.yield %transfer : !ttl.transfer_handle<read>
  }
  ttl.wait %selected : !ttl.transfer_handle<read>
  ttl.cb_push %storage : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  return
}

// A transfer without a wait leaves its payload access incomplete.
// CHECK-LABEL: func.func @missing_completion_wait
// CHECK: ttkernel.noc_async_read_tile
// CHECK: ttkernel.cb_push_back(%{{.*}}, %{{.*}}) :
// CHECK: return
func.func @missing_completion_wait(
    %source: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>)
    attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0],
                ttl.kernel_thread = #ttkernel.thread<noc>} {
  %zero = arith.constant 0 : index
  %storage = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %reserved = ttl.cb_reserve %storage
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %slice = ttl.tensor_slice %source[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
        -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
  %transfer = ttl.copy %slice, %storage
      : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> !ttl.transfer_handle<read>
  ttl.cb_push %storage : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  return
}

// An external DFB user can issue payload accesses not covered by the wait.
// CHECK-LABEL: func.func @external_storage_user
// CHECK: ttkernel.noc_async_read_barrier
// CHECK: ttkernel.opaque_call
// CHECK: ttkernel.cb_push_back(%{{.*}}, %{{.*}}) :
// CHECK: return
func.func @external_storage_user(
    %source: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>)
    attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0],
                ttl.kernel_thread = #ttkernel.thread<noc>} {
  %zero = arith.constant 0 : index
  %storage = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %reserved = ttl.cb_reserve %storage
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %slice = ttl.tensor_slice %source[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
        -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
  %transfer = ttl.copy %slice, %storage
      : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> !ttl.transfer_handle<read>
  ttl.wait %transfer : !ttl.transfer_handle<read>
  ttl.opaque_call "inspect_dfb"
      template_args [#ttl.external_template_arg<dfb_descriptor, 0>]
      template_dfbs(%storage : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      () {header = "inspect_dfb.hpp"} : () -> ()
  ttl.cb_push %storage : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  return
}

}
