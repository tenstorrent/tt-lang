// Node (1, 0) pushes a block before two reconfigurations that do not discard
// state and pops it after both, and its lifecycle is not proven complete, so the
// allocation group is trusted only with unsafe-assume-allocation-groups. Node
// (0, 0) uses a larger DFB of the same group between the boundaries, so the
// configuration entered at the first boundary has a different geometry and is
// installed on node (1, 0). The configuration entered at the second boundary
// differs from that installed descriptor, so it is installed on node (1, 0) too.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true unsafe-assume-allocation-groups=true})' | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true unsafe-assume-allocation-groups=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEBUG

// CHECK: ttl.dfb_reconfiguration_plan = {boundary_ordinals = array<i64: 0, 1>, dfbs = [{configurations = [{block_count = 2 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32, page_size = 2048 : i32, storage_segments = [{nodes = {{\[\[}}0, 0], [1, 0{{\]\]}}}]}, {block_count = 4 : i32, element_type = !ttcore.tile<32x32, bf16>, entry_reconfiguration = 0 : i64, num_tiles = 1 : i32, page_size = 2048 : i32, storage_segments = [{nodes = {{\[\[}}0, 0], [1, 0{{\]\]}}}]}, {block_count = 2 : i32, element_type = !ttcore.tile<32x32, bf16>, entry_reconfiguration = 1 : i64, num_tiles = 1 : i32, page_size = 2048 : i32, storage_segments = [{nodes = {{\[\[}}0, 0], [1, 0{{\]\]}}}]}], dfb_index = 0 : i32}]}
// DEBUG: node (1,0) lifecycle_completion=incomplete-use-order

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#layout = #ttl.layout<shape = [32, 128], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 2], memory = interleaved>
#first = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>
#second = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer]>

module attributes {ttl.launch_grid = [2, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #compute} {
    ttl.dfb_reconfiguration #first
    ttl.dfb_reconfiguration #second
    return
  }
  func.func @reader(%arg0: tensor<1x4x!ttcore.tile<32x32, bf16>, #layout>) attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [0 : i32], ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #reader, ttl.noc_index = 0 : i32} {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %0 = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %1 = ttl.bind_cb{cb_index = 1, block_count = 4} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %2 = ttl.cb_reserve %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %3 = ttl.attach_cb %2, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %4 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
    %5 = ttl.copy %4, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %5 : !ttl.transfer_handle<read>
    ttl.cb_push %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.dfb_reconfiguration #first
    %6 = ttl.core_x : index
    %7 = arith.index_cast %c0_i64 : i64 to index
    %8 = arith.cmpi eq, %6, %7 : index
    scf.if %8 {
      %11 = ttl.cb_reserve %1 : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %12 = ttl.attach_cb %11, %1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %13 = ttl.tensor_slice %arg0[%c0, %c1] : tensor<1x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
      %14 = ttl.copy %13, %1 : (tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> !ttl.transfer_handle<read>
      ttl.wait %14 : !ttl.transfer_handle<read>
      ttl.cb_push %1 : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    }
    ttl.dfb_reconfiguration #second
    %9 = arith.index_cast %c0_i64 : i64 to index
    %10 = arith.cmpi eq, %6, %9 : index
    scf.if %10 {
      %11 = ttl.cb_reserve %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %12 = ttl.attach_cb %11, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %13 = ttl.tensor_slice %arg0[%c0, %c2] : tensor<1x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
      %14 = ttl.copy %13, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
      ttl.wait %14 : !ttl.transfer_handle<read>
      ttl.cb_push %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    return
  }
  func.func @writer(%arg0: tensor<1x4x!ttcore.tile<32x32, bf16>, #layout>) attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [1 : i32], ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #writer, ttl.noc_index = 1 : i32} {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %0 = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %1 = ttl.bind_cb{cb_index = 1, block_count = 4} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %2 = ttl.core_x : index
    %3 = arith.index_cast %c0_i64 : i64 to index
    %4 = arith.cmpi eq, %2, %3 : index
    scf.if %4 {
      %11 = ttl.cb_wait %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %12 = ttl.attach_cb %11, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %13 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
      %14 = ttl.copy %0, %13 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>) -> !ttl.transfer_handle<write>
      ttl.wait %14 : !ttl.transfer_handle<write>
      ttl.cb_pop %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    ttl.dfb_reconfiguration #first
    %5 = arith.index_cast %c0_i64 : i64 to index
    %6 = arith.cmpi eq, %2, %5 : index
    scf.if %6 {
      %11 = ttl.cb_wait %1 : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %12 = ttl.attach_cb %11, %1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %13 = ttl.tensor_slice %arg0[%c0, %c1] : tensor<1x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
      %14 = ttl.copy %1, %13 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>, tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>) -> !ttl.transfer_handle<write>
      ttl.wait %14 : !ttl.transfer_handle<write>
      ttl.cb_pop %1 : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    }
    ttl.dfb_reconfiguration #second
    %7 = ttl.cb_wait %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %8 = ttl.attach_cb %7, %0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %9 = arith.index_cast %c0_i64 : i64 to index
    %10 = arith.cmpi eq, %2, %9 : index
    scf.if %10 {
      %11 = ttl.tensor_slice %arg0[%c0, %c2] : tensor<1x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
      %12 = ttl.copy %0, %11 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>) -> !ttl.transfer_handle<write>
      ttl.wait %12 : !ttl.transfer_handle<write>
    } else {
      %11 = ttl.tensor_slice %arg0[%c0, %c3] : tensor<1x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
      %12 = ttl.copy %0, %11 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>) -> !ttl.transfer_handle<write>
      ttl.wait %12 : !ttl.transfer_handle<write>
    }
    ttl.cb_pop %0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
