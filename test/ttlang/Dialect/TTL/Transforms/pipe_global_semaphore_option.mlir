// RUN: ttlang-opt %s --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-global-semaphores-only=false})' | FileCheck %s --check-prefix=LOCAL
// Global-only mode must not lower any counter to a local semaphore lookup.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-global-semaphores-only=true})' | FileCheck %s --check-prefix=GLOBAL --implicit-check-not=ttkernel.get_semaphore

// Summary: Verifies that the PipeNet counter-storage option preserves the
// allocation plan while selecting local or GlobalSemaphore storage.

// The default uses one local completion counter and one local sender-ready
// counter. Global-only mode preserves their synchronization data flow while
// leaving the local semaphore count at zero.
// LOCAL-LABEL: module attributes
// LOCAL-NOT: ttl.pipe_global_semaphore_count
// LOCAL-SAME: ttl.pipe_sync_semaphore_count = 2 : i64
// LOCAL-LABEL: func.func @select_pipe_counter_storage
// LOCAL-DAG: %[[LOCAL_COMPLETION_INDEX:.*]] = arith.constant 0 : index
// LOCAL-DAG: %[[LOCAL_READY_INDEX:.*]] = arith.constant 1 : index
// LOCAL: %[[LOCAL_POST_COUNTER:.*]] = ttkernel.get_semaphore(%[[LOCAL_READY_INDEX]])
// LOCAL: %[[LOCAL_POST_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, %[[LOCAL_POST_COUNTER]], {{.*}})
// LOCAL-NEXT: ttkernel.noc_semaphore_inc(%[[LOCAL_POST_NOC]]
// LOCAL: %[[LOCAL_READY_COUNTER:.*]] = ttkernel.get_semaphore(%[[LOCAL_READY_INDEX]])
// LOCAL-NEXT: %[[LOCAL_READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[LOCAL_READY_COUNTER]])
// LOCAL-NEXT: ttkernel.experimental.semaphore_wait(%[[LOCAL_READY_PTR]]
// LOCAL-NEXT: ttkernel.noc_semaphore_set(%[[LOCAL_READY_PTR]]
// LOCAL: %[[LOCAL_COMPLETION_COUNTER:.*]] = ttkernel.get_semaphore(%[[LOCAL_COMPLETION_INDEX]])
// LOCAL: %[[LOCAL_COMPLETION_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, %[[LOCAL_COMPLETION_COUNTER]], {{.*}})
// LOCAL-NEXT: ttkernel.noc_semaphore_inc(%[[LOCAL_COMPLETION_NOC]]
// LOCAL: %[[LOCAL_WAIT_COUNTER:.*]] = ttkernel.get_semaphore(%[[LOCAL_COMPLETION_INDEX]])
// LOCAL-NEXT: %[[LOCAL_WAIT_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[LOCAL_WAIT_COUNTER]])
// LOCAL: ttkernel.experimental.semaphore_wait_min(%[[LOCAL_WAIT_PTR]]

// GLOBAL-LABEL: module attributes
// GLOBAL-SAME: ttl.pipe_global_semaphore_count = 2 : i64
// GLOBAL-SAME: ttl.pipe_sync_semaphore_count = 0 : i64
// GLOBAL-LABEL: func.func @select_pipe_counter_storage
// GLOBAL-DAG: %[[GLOBAL_COMPLETION_INDEX:.*]] = arith.constant 1 : index
// GLOBAL-DAG: %[[GLOBAL_READY_INDEX:.*]] = arith.constant 2 : index
// GLOBAL: %[[GLOBAL_POST_COUNTER:.*]] = ttkernel.get_common_arg_val(%[[GLOBAL_READY_INDEX]])
// GLOBAL: %[[GLOBAL_POST_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, %[[GLOBAL_POST_COUNTER]], {{.*}})
// GLOBAL-NEXT: ttkernel.noc_semaphore_inc(%[[GLOBAL_POST_NOC]]
// GLOBAL: %[[GLOBAL_READY_COUNTER:.*]] = ttkernel.get_common_arg_val(%[[GLOBAL_READY_INDEX]])
// GLOBAL-NEXT: %[[GLOBAL_READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[GLOBAL_READY_COUNTER]])
// GLOBAL-NEXT: ttkernel.experimental.semaphore_wait(%[[GLOBAL_READY_PTR]]
// GLOBAL-NEXT: ttkernel.noc_semaphore_set(%[[GLOBAL_READY_PTR]]
// GLOBAL: %[[GLOBAL_COMPLETION_COUNTER:.*]] = ttkernel.get_common_arg_val(%[[GLOBAL_COMPLETION_INDEX]])
// GLOBAL: %[[GLOBAL_COMPLETION_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, %[[GLOBAL_COMPLETION_COUNTER]], {{.*}})
// GLOBAL-NEXT: ttkernel.noc_semaphore_inc(%[[GLOBAL_COMPLETION_NOC]]
// GLOBAL: %[[GLOBAL_WAIT_COUNTER:.*]] = ttkernel.get_common_arg_val(%[[GLOBAL_COMPLETION_INDEX]])
// GLOBAL-NEXT: %[[GLOBAL_WAIT_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[GLOBAL_WAIT_COUNTER]])
// GLOBAL: ttkernel.experimental.semaphore_wait_min(%[[GLOBAL_WAIT_PTR]]

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @select_pipe_counter_storage()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %reserved = ttl.cb_reserve %dst
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %receive = ttl.copy %pipe, %reserved
        : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send = ttl.copy %src, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    ttl.wait %receive : !ttl.transfer_handle
    ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    func.return
  }
}
