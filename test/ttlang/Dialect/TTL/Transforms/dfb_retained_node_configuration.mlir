// Node (0, 0) completes one lifecycle on each side of the boundary, so the plan
// enters a configuration there. Node (1, 0) pushes a block before the boundary
// and pops it after the boundary, and its lifecycle is not proven complete. The
// runtime resets the pointers and counters of every node that the entered
// configuration covers.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEBUG

// Without state discard, node (1, 0) retains its block across the boundary, so
// the configuration is installed on node (0, 0) only.
// CHECK: ttl.dfb_reconfiguration_plan = {boundary_ordinals = array<i64: 0>, dfbs = [{configurations = [{block_count = 2 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32, page_size = 2048 : i32, storage_segments = [{nodes = {{\[\[}}0, 0], [1, 0{{\]\]}}}]}, {block_count = 2 : i32, element_type = !ttcore.tile<32x32, bf16>, entry_reconfiguration = 0 : i64, num_tiles = 1 : i32, page_size = 2048 : i32, storage_segments = [{nodes = {{\[\[}}0, 0{{\]\]}}}]}], dfb_index = 0 : i32}]}
// DEBUG: node (0,0) lifecycle_completion=complete
// DEBUG: node (1,0) lifecycle_completion=incomplete-use-order

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#layout = #ttl.layout<shape = [32, 128], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 2], memory = interleaved>
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

module attributes {ttl.launch_grid = [2, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #compute} {
    ttl.dfb_reconfiguration #boundary
    return
  }
  func.func @reader(%arg0: tensor<1x4x!ttcore.tile<32x32, bf16>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0 : i32], ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #reader, ttl.noc_index = 0 : i32} {
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %0 = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %1 = ttl.cb_reserve %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %2 = ttl.attach_cb %1, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %3 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
    %4 = ttl.copy %3, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %4 : !ttl.transfer_handle<read>
    ttl.cb_push %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.dfb_reconfiguration #boundary
    %5 = ttl.core_x : index
    %6 = arith.index_cast %c0_i64 : i64 to index
    %7 = arith.cmpi eq, %5, %6 : index
    scf.if %7 {
      %8 = ttl.cb_reserve %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %9 = ttl.attach_cb %8, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %10 = ttl.tensor_slice %arg0[%c0, %c1] : tensor<1x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
      %11 = ttl.copy %10, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
      ttl.wait %11 : !ttl.transfer_handle<read>
      ttl.cb_push %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    return
  }
  func.func @writer(%arg0: tensor<1x4x!ttcore.tile<32x32, bf16>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [1 : i32], ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #writer, ttl.noc_index = 1 : i32} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %0 = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %1 = ttl.core_x : index
    %2 = arith.index_cast %c0_i64 : i64 to index
    %3 = arith.cmpi eq, %1, %2 : index
    scf.if %3 {
      %8 = ttl.cb_wait %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %9 = ttl.attach_cb %8, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %10 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
      %11 = ttl.copy %0, %10 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>) -> !ttl.transfer_handle<write>
      ttl.wait %11 : !ttl.transfer_handle<write>
      ttl.cb_pop %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    ttl.dfb_reconfiguration #boundary
    %4 = ttl.cb_wait %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %5 = ttl.attach_cb %4, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %6 = ttl.tensor_slice %arg0[%c0, %c1] : tensor<1x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
    %7 = ttl.copy %0, %6 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>) -> !ttl.transfer_handle<write>
    ttl.wait %7 : !ttl.transfer_handle<write>
    ttl.cb_pop %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// With state discard, the boundary may end node (1, 0)'s lifecycle, so the
// configuration is installed on both nodes.
// CHECK: ttl.dfb_reconfiguration_plan = {boundary_ordinals = array<i64: 0>, dfbs = [{configurations = [{block_count = 2 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32, page_size = 2048 : i32, storage_segments = [{nodes = {{\[\[}}0, 0], [1, 0{{\]\]}}}]}, {block_count = 2 : i32, element_type = !ttcore.tile<32x32, bf16>, entry_reconfiguration = 0 : i64, num_tiles = 1 : i32, page_size = 2048 : i32, storage_segments = [{nodes = {{\[\[}}0, 0], [1, 0{{\]\]}}}]}], dfb_index = 0 : i32}]}
// DEBUG: node (0,0) lifecycle_completion=complete
// DEBUG: node (1,0) lifecycle_completion=incomplete-use-order

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#layout = #ttl.layout<shape = [32, 128], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 2], memory = interleaved>
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer], discard_dfb_state = true>

module attributes {ttl.launch_grid = [2, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #compute} {
    ttl.dfb_reconfiguration #boundary
    return
  }
  func.func @reader(%arg0: tensor<1x4x!ttcore.tile<32x32, bf16>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0 : i32], ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #reader, ttl.noc_index = 0 : i32} {
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %0 = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %1 = ttl.cb_reserve %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %2 = ttl.attach_cb %1, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %3 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
    %4 = ttl.copy %3, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %4 : !ttl.transfer_handle<read>
    ttl.cb_push %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.dfb_reconfiguration #boundary
    %5 = ttl.core_x : index
    %6 = arith.index_cast %c0_i64 : i64 to index
    %7 = arith.cmpi eq, %5, %6 : index
    scf.if %7 {
      %8 = ttl.cb_reserve %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %9 = ttl.attach_cb %8, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %10 = ttl.tensor_slice %arg0[%c0, %c1] : tensor<1x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
      %11 = ttl.copy %10, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
      ttl.wait %11 : !ttl.transfer_handle<read>
      ttl.cb_push %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    return
  }
  func.func @writer(%arg0: tensor<1x4x!ttcore.tile<32x32, bf16>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [1 : i32], ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #writer, ttl.noc_index = 1 : i32} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %0 = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %1 = ttl.core_x : index
    %2 = arith.index_cast %c0_i64 : i64 to index
    %3 = arith.cmpi eq, %1, %2 : index
    scf.if %3 {
      %8 = ttl.cb_wait %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %9 = ttl.attach_cb %8, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %10 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
      %11 = ttl.copy %0, %10 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>) -> !ttl.transfer_handle<write>
      ttl.wait %11 : !ttl.transfer_handle<write>
      ttl.cb_pop %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    ttl.dfb_reconfiguration #boundary
    %4 = ttl.cb_wait %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %5 = ttl.attach_cb %4, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %6 = ttl.tensor_slice %arg0[%c0, %c1] : tensor<1x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
    %7 = ttl.copy %0, %6 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>) -> !ttl.transfer_handle<write>
    ttl.wait %7 : !ttl.transfer_handle<write>
    ttl.cb_pop %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
