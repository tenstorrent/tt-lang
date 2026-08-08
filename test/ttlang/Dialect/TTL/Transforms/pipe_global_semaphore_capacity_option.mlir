// RUN: ttlang-opt %s --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=true pipe-global-semaphores-only=false})' | FileCheck %s --check-prefix=LOCAL
// Global-only mode must not lower any capacity or completion counter to a local semaphore lookup.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=true pipe-global-semaphores-only=true})' | FileCheck %s --check-prefix=GLOBAL --implicit-check-not=ttkernel.get_semaphore

// Summary: Verifies that global-only counter allocation applies to the
// completion and capacity counters used by CA/CC synchronization.

// The local-first configuration uses one local completion counter and one
// local capacity counter.
// LOCAL-LABEL: module attributes
// LOCAL-NOT: ttl.pipe_global_semaphore_count
// LOCAL-SAME: ttl.pipe_sync_semaphore_count = 2 : i64
// LOCAL-LABEL: func.func @select_capacity_counter_storage
// LOCAL-DAG: %[[LOCAL_COMPLETION_INDEX:.*]] = arith.constant 0 : index
// LOCAL-DAG: %[[LOCAL_CAPACITY_INDEX:.*]] = arith.constant 1 : index
// LOCAL: %[[LOCAL_CAPACITY_INIT:.*]] = ttkernel.get_semaphore(%[[LOCAL_CAPACITY_INDEX]])
// LOCAL-NEXT: %[[LOCAL_CAPACITY_INIT_PTR:.*]] = ttkernel.reinterpret_cast(%[[LOCAL_CAPACITY_INIT]])
// LOCAL-NEXT: ttkernel.noc_semaphore_set(%[[LOCAL_CAPACITY_INIT_PTR]],
// LOCAL: %[[LOCAL_COMPLETION_WAIT:.*]] = ttkernel.get_semaphore(%[[LOCAL_COMPLETION_INDEX]])
// LOCAL-NEXT: %[[LOCAL_COMPLETION_WAIT_PTR:.*]] = ttkernel.reinterpret_cast(%[[LOCAL_COMPLETION_WAIT]])
// LOCAL-NEXT: ttkernel.experimental.semaphore_wait_min(%[[LOCAL_COMPLETION_WAIT_PTR]],
// LOCAL: ttkernel.cb_pop_front
// LOCAL-NEXT: %[[LOCAL_CAPACITY_RELEASE:.*]] = ttkernel.get_semaphore(%[[LOCAL_CAPACITY_INDEX]])
// LOCAL: %[[LOCAL_CAPACITY_RELEASE_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, %[[LOCAL_CAPACITY_RELEASE]],
// LOCAL-NEXT: ttkernel.noc_semaphore_inc(%[[LOCAL_CAPACITY_RELEASE_NOC]],
// LOCAL: %[[LOCAL_CAPACITY_ACQUIRE:.*]] = ttkernel.get_semaphore(%[[LOCAL_CAPACITY_INDEX]])
// LOCAL-NEXT: %[[LOCAL_CAPACITY_ACQUIRE_PTR:.*]] = ttkernel.reinterpret_cast(%[[LOCAL_CAPACITY_ACQUIRE]])
// LOCAL: ttkernel.experimental.semaphore_wait_min(%[[LOCAL_CAPACITY_ACQUIRE_PTR]],
// LOCAL: %[[LOCAL_COMPLETION_SIGNAL:.*]] = ttkernel.get_semaphore(%[[LOCAL_COMPLETION_INDEX]])
// LOCAL: %[[LOCAL_COMPLETION_SIGNAL_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, %[[LOCAL_COMPLETION_SIGNAL]],
// LOCAL-NEXT: ttkernel.noc_semaphore_inc(%[[LOCAL_COMPLETION_SIGNAL_NOC]],

// Global-only mode preserves the same two-counter plan and uses no local ids.
// GLOBAL-LABEL: module attributes
// GLOBAL-SAME: ttl.pipe_global_semaphore_count = 2 : i64
// GLOBAL-SAME: ttl.pipe_sync_semaphore_count = 0 : i64
// GLOBAL-LABEL: func.func @select_capacity_counter_storage
// GLOBAL-DAG: %[[GLOBAL_COMPLETION_INDEX:.*]] = arith.constant 1 : index
// GLOBAL-DAG: %[[GLOBAL_CAPACITY_INDEX:.*]] = arith.constant 2 : index
// GLOBAL: %[[GLOBAL_CAPACITY_INIT:.*]] = ttkernel.get_common_arg_val(%[[GLOBAL_CAPACITY_INDEX]])
// GLOBAL-NEXT: %[[GLOBAL_CAPACITY_INIT_PTR:.*]] = ttkernel.reinterpret_cast(%[[GLOBAL_CAPACITY_INIT]])
// GLOBAL-NEXT: ttkernel.noc_semaphore_set(%[[GLOBAL_CAPACITY_INIT_PTR]],
// GLOBAL: %[[GLOBAL_COMPLETION_WAIT:.*]] = ttkernel.get_common_arg_val(%[[GLOBAL_COMPLETION_INDEX]])
// GLOBAL-NEXT: %[[GLOBAL_COMPLETION_WAIT_PTR:.*]] = ttkernel.reinterpret_cast(%[[GLOBAL_COMPLETION_WAIT]])
// GLOBAL-NEXT: ttkernel.experimental.semaphore_wait_min(%[[GLOBAL_COMPLETION_WAIT_PTR]],
// GLOBAL: ttkernel.cb_pop_front
// GLOBAL-NEXT: %[[GLOBAL_CAPACITY_RELEASE:.*]] = ttkernel.get_common_arg_val(%[[GLOBAL_CAPACITY_INDEX]])
// GLOBAL: %[[GLOBAL_CAPACITY_RELEASE_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, %[[GLOBAL_CAPACITY_RELEASE]],
// GLOBAL-NEXT: ttkernel.noc_semaphore_inc(%[[GLOBAL_CAPACITY_RELEASE_NOC]],
// GLOBAL: %[[GLOBAL_CAPACITY_ACQUIRE:.*]] = ttkernel.get_common_arg_val(%[[GLOBAL_CAPACITY_INDEX]])
// GLOBAL-NEXT: %[[GLOBAL_CAPACITY_ACQUIRE_PTR:.*]] = ttkernel.reinterpret_cast(%[[GLOBAL_CAPACITY_ACQUIRE]])
// GLOBAL: ttkernel.experimental.semaphore_wait_min(%[[GLOBAL_CAPACITY_ACQUIRE_PTR]],
// GLOBAL: %[[GLOBAL_COMPLETION_SIGNAL:.*]] = ttkernel.get_common_arg_val(%[[GLOBAL_COMPLETION_INDEX]])
// GLOBAL: %[[GLOBAL_COMPLETION_SIGNAL_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, %[[GLOBAL_COMPLETION_SIGNAL]],
// GLOBAL-NEXT: ttkernel.noc_semaphore_inc(%[[GLOBAL_COMPLETION_SIGNAL_NOC]],

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @select_capacity_counter_storage()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe
        {expectedReceivers = 1 : i64,
         kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %reserved = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %receive = ttl.pipe_transfer.post %transfer, %reserved
          : (!ttl.pipe_transfer,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %receive : !ttl.pipe_token<net 0>
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      %ready = ttl.cb_wait %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.pipe_transfer.send %transfer, %src
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}
