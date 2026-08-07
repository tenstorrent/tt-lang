// RUN: ttlang-opt %s --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-global-semaphores-only=false})' | FileCheck %s --check-prefix=LOCAL
// Global-only mode must not lower any PipeNet counter to a local semaphore.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-global-semaphores-only=true})' | FileCheck %s --check-prefix=GLOBAL --implicit-check-not=ttkernel.get_semaphore

// Summary: Verifies counter-storage selection for a module containing both
// intra-device and fabric PipeNet transfers.

// The default allocates the intra-device completion and ready counters from
// local semaphore ids. Fabric completion requires GlobalSemaphore storage.
// LOCAL-LABEL: module attributes
// LOCAL-SAME: ttl.pipe_global_semaphore_count = 1 : i64
// LOCAL-SAME: ttl.pipe_sync_semaphore_count = 2 : i64
// LOCAL-LABEL: func.func @local_transfer
// LOCAL-DAG: %[[LOCAL_COMPLETION_INDEX:.*]] = arith.constant 0 : index
// LOCAL-DAG: %[[LOCAL_READY_INDEX:.*]] = arith.constant 1 : index
// LOCAL: %[[LOCAL_READY:.*]] = ttkernel.get_semaphore(%[[LOCAL_READY_INDEX]])
// LOCAL: %[[LOCAL_READY_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, %[[LOCAL_READY]], {{.*}})
// LOCAL-NEXT: ttkernel.noc_semaphore_inc(%[[LOCAL_READY_NOC]]
// LOCAL: %[[LOCAL_COMPLETION:.*]] = ttkernel.get_semaphore(%[[LOCAL_COMPLETION_INDEX]])
// LOCAL: %[[LOCAL_COMPLETION_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, %[[LOCAL_COMPLETION]], {{.*}})
// LOCAL-NEXT: ttkernel.noc_semaphore_inc(%[[LOCAL_COMPLETION_NOC]]
// LOCAL-LABEL: func.func @fabric_sender
// LOCAL-DAG: %[[LOCAL_FABRIC_DONE_INDEX:.*]] = arith.constant 1 : index
// LOCAL: %[[LOCAL_FABRIC_DONE:.*]] = ttkernel.get_common_arg_val(%[[LOCAL_FABRIC_DONE_INDEX]])
// LOCAL: %[[LOCAL_REMOTE_DONE:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[LOCAL_FABRIC_DONE]], {{.*}})
// LOCAL: ttkernel.routing_plane.fused_write_atomic_inc({{.*}}, %[[LOCAL_REMOTE_DONE]], {{.*}})
// LOCAL-NOT: ttkernel.get_semaphore
// LOCAL-LABEL: func.func @fabric_receiver
// LOCAL-DAG: %[[LOCAL_FABRIC_WAIT_INDEX:.*]] = arith.constant 0 : index
// LOCAL: %[[LOCAL_FABRIC_WAIT:.*]] = ttkernel.get_common_arg_val(%[[LOCAL_FABRIC_WAIT_INDEX]])
// LOCAL-NEXT: %[[LOCAL_FABRIC_WAIT_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[LOCAL_FABRIC_WAIT]])
// LOCAL: ttkernel.experimental.semaphore_wait_min(%[[LOCAL_FABRIC_WAIT_PTR]]
// LOCAL-NOT: ttkernel.get_semaphore

// Global-only mode moves the intra-device counters to GlobalSemaphore storage;
// the fabric completion counter remains global.
// GLOBAL-LABEL: module attributes
// GLOBAL-SAME: ttl.pipe_global_semaphore_count = 3 : i64
// GLOBAL-SAME: ttl.pipe_sync_semaphore_count = 0 : i64
// GLOBAL-LABEL: func.func @local_transfer
// GLOBAL-DAG: %[[GLOBAL_COMPLETION_INDEX:.*]] = arith.constant 1 : index
// GLOBAL-DAG: %[[GLOBAL_READY_INDEX:.*]] = arith.constant 3 : index
// GLOBAL: %[[GLOBAL_READY:.*]] = ttkernel.get_common_arg_val(%[[GLOBAL_READY_INDEX]])
// GLOBAL: %[[GLOBAL_READY_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, %[[GLOBAL_READY]], {{.*}})
// GLOBAL-NEXT: ttkernel.noc_semaphore_inc(%[[GLOBAL_READY_NOC]]
// GLOBAL: %[[GLOBAL_COMPLETION:.*]] = ttkernel.get_common_arg_val(%[[GLOBAL_COMPLETION_INDEX]])
// GLOBAL: %[[GLOBAL_COMPLETION_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, %[[GLOBAL_COMPLETION]], {{.*}})
// GLOBAL-NEXT: ttkernel.noc_semaphore_inc(%[[GLOBAL_COMPLETION_NOC]]
// GLOBAL-LABEL: func.func @fabric_sender
// GLOBAL-DAG: %[[GLOBAL_FABRIC_DONE_INDEX:.*]] = arith.constant 2 : index
// GLOBAL: %[[GLOBAL_FABRIC_DONE:.*]] = ttkernel.get_common_arg_val(%[[GLOBAL_FABRIC_DONE_INDEX]])
// GLOBAL: %[[GLOBAL_REMOTE_DONE:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[GLOBAL_FABRIC_DONE]], {{.*}})
// GLOBAL: ttkernel.routing_plane.fused_write_atomic_inc({{.*}}, %[[GLOBAL_REMOTE_DONE]], {{.*}})
// GLOBAL-LABEL: func.func @fabric_receiver
// GLOBAL-DAG: %[[GLOBAL_FABRIC_WAIT_INDEX:.*]] = arith.constant 1 : index
// GLOBAL: %[[GLOBAL_FABRIC_WAIT:.*]] = ttkernel.get_common_arg_val(%[[GLOBAL_FABRIC_WAIT_INDEX]])
// GLOBAL-NEXT: %[[GLOBAL_FABRIC_WAIT_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[GLOBAL_FABRIC_WAIT]])
// GLOBAL: ttkernel.experimental.semaphore_wait_min(%[[GLOBAL_FABRIC_WAIT_PTR]]

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @local_transfer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
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

  func.func @fabric_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 2, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 37 {
      deviceTransfer = #ttl.device_transfer<
          domain = <components = <name = "device", extent = [1, 4]>>,
          edge = <source = <coordinates = [0, 2]>,
                  destination = <coordinates = [0, 0]>>>
    } : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 37>
    %send = ttl.copy %src, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 37>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    func.return
  }

  func.func @fabric_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dst = ttl.bind_cb {cb_index = 3, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 37 {
      deviceTransfer = #ttl.device_transfer<
          domain = <components = <name = "device", extent = [1, 4]>>,
          edge = <source = <coordinates = [0, 2]>,
                  destination = <coordinates = [0, 0]>>>
    } : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 37>
    %reserved = ttl.cb_reserve %dst
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %receive = ttl.copy %pipe, %reserved
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 37>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %receive : !ttl.transfer_handle
    ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    func.return
  }
}
