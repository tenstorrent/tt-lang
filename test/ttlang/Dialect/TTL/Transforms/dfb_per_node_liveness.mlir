// Tests launch-node-local DFB lifetimes and hardware pointer ownership.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s

// DFBs used on disjoint launch nodes may share without a global lifetime order.

// CHECK: module attributes {ttl.dfb_allocations = [{{.*}}dfb_index = 0 : i32{{.*}}]}
// CHECK-LABEL: func.func @disjoint_node_producer
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 10 : index}
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 11 : index}
// CHECK-LABEL: func.func @disjoint_node_consumer
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 10 : index}
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 11 : index}

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @disjoint_node_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %first_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 10 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 11 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %is_node_zero = arith.cmpi eq, %core_x, %zero : index
    %is_node_one = arith.cmpi eq, %core_x, %one : index
    scf.if %is_node_zero {
      %first_slot = ttl.cb_reserve %first_dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      ttl.cb_push %first_dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    }
    scf.if %is_node_one {
      %second_slot = ttl.cb_reserve %second_dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      ttl.cb_push %second_dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    }
    return
  }

  func.func @disjoint_node_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %first_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 10 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 11 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %is_node_zero = arith.cmpi eq, %core_x, %zero : index
    %is_node_one = arith.cmpi eq, %core_x, %one : index
    scf.if %is_node_zero {
      %first_value = ttl.cb_wait %first_dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      ttl.cb_pop %first_dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    }
    scf.if %is_node_one {
      %second_value = ttl.cb_wait %second_dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      ttl.cb_pop %second_dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    }
    return
  }
}

// -----

// Distinct function symbols may share when their per-node pointer processors
// match and both producer and consumer entries follow a quiescent handoff.

// CHECK: module attributes {ttl.dfb_allocations = [{{.*}}dfb_index = 0 : i32{{.*}}, {{.*}}dfb_index = 1 : i32{{.*}}, {{.*}}dfb_index = 2 : i32{{.*}}]}
// CHECK-LABEL: func.func @first_owner_producer
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 20 : index}
// CHECK-LABEL: func.func @first_owner_consumer
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 20 : index}
// CHECK-LABEL: func.func @second_owner_producer
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 23 : index}
// CHECK-LABEL: func.func @second_owner_consumer
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 23 : index}

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @first_owner_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %first_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 20 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %first_slot = ttl.cb_reserve %first_dfb
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %first_dfb
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    return
  }

  func.func @first_owner_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %first_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 20 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %producer_ack = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 21 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, f32>, 2>
    %consumer_ack = ttl.bind_cb {cb_index = 2, block_count = 2}
        {dfb_id = 22 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, f32>, 2>
    %first_value = ttl.cb_wait %first_dfb
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %first_dfb
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %producer_ack_slot = ttl.cb_reserve %producer_ack
        : <[1, 1], !ttcore.tile<1x16, f32>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, f32>>
    ttl.cb_push %producer_ack
        : <[1, 1], !ttcore.tile<1x16, f32>, 2>
    %consumer_ack_slot = ttl.cb_reserve %consumer_ack
        : <[1, 1], !ttcore.tile<1x16, f32>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, f32>>
    ttl.cb_push %consumer_ack
        : <[1, 1], !ttcore.tile<1x16, f32>, 2>
    return
  }

  func.func @second_owner_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %producer_ack = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 21 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, f32>, 2>
    %second_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
        {dfb_id = 23 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %producer_ack_value = ttl.cb_wait %producer_ack
        : <[1, 1], !ttcore.tile<1x16, f32>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, f32>>
    ttl.cb_pop %producer_ack
        : <[1, 1], !ttcore.tile<1x16, f32>, 2>
    %second_slot = ttl.cb_reserve %second_dfb
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %second_dfb
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    return
  }

  func.func @second_owner_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %consumer_ack = ttl.bind_cb {cb_index = 2, block_count = 2}
        {dfb_id = 22 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, f32>, 2>
    %second_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
        {dfb_id = 23 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %consumer_ack_value = ttl.cb_wait %consumer_ack
        : <[1, 1], !ttcore.tile<1x16, f32>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, f32>>
    ttl.cb_pop %consumer_ack
        : <[1, 1], !ttcore.tile<1x16, f32>, 2>
    %second_value = ttl.cb_wait %second_dfb
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %second_dfb
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    return
  }
}

// -----

// Quiescent lifetimes with different transaction sizes retain separate ring
// pointer progressions and therefore cannot share a physical index.

// CHECK: module attributes {ttl.dfb_allocations = [{{.*}}dfb_index = 0 : i32{{.*}}, {{.*}}dfb_index = 1 : i32{{.*}}]}
// CHECK-LABEL: func.func @different_transaction_sizes
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 30 : index}
// CHECK: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 31 : index}

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @different_transaction_sizes()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %single_tile_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 30 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %two_tile_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 31 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %single_tile_slot = ttl.cb_reserve %single_tile_dfb
        {num_tiles = 1 : i64}
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %single_tile_dfb {num_tiles = 1 : i64}
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %single_tile_value = ttl.cb_wait %single_tile_dfb
        {num_tiles = 1 : i64}
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %single_tile_dfb {num_tiles = 1 : i64}
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %two_tile_slot = ttl.cb_reserve %two_tile_dfb
        {num_tiles = 2 : i64}
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x2x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %two_tile_dfb {num_tiles = 2 : i64}
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %two_tile_value = ttl.cb_wait %two_tile_dfb
        {num_tiles = 2 : i64}
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x2x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %two_tile_dfb {num_tiles = 2 : i64}
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    return
  }
}
