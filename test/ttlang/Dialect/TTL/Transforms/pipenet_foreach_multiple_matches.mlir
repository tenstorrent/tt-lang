// Summary: Verifies distinct resources for every matching PipeNet record.
// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-form-pipe-transports,ttl-finalize-dfb-indices,convert-ttl-to-ttkernel)' | FileCheck %s
// RUN: ttlang-opt %s -ttl-verify-pipenet-guards

// Verifies that one table-driven protocol operation receives distinct
// resources for every matching record, including identical records.
// Running static transport planning before conversion must preserve the same
// selected-record address and synchronization tables.

// CHECK-NOT: ttl.pipe_conservative_l1_bytes
// CHECK: ttl.launch_grid = array<i64: 2, 5>, ttl.pipe_sram_scratch_bytes = 32 : i64, ttl.pipe_sync_semaphore_count = 8 : i64

module attributes {ttl.launch_grid = array<i64: 2, 5>} {

// CHECK-LABEL: func.func @gather_receiver
// Six distinct completion counters cover the six matching records.
// CHECK: %[[COUNTERS:.*]] = memref.alloca() : memref<6xi32>
// CHECK: scf.for %[[INDEX:.*]] =
// CHECK: ttkernel.experimental.constant_table_lookup %[[INDEX]], [6, 6, 6, 6, 6, 7] : index
// CHECK: %[[PROGRESS_INDEX:.*]] = ttkernel.experimental.constant_table_lookup %[[INDEX]], [0, 2, 3, 4, 5, 1] : index
// CHECK: memref.load %[[COUNTERS]][%[[PROGRESS_INDEX]]] : memref<6xi32>
// CHECK: ttkernel.experimental.semaphore_wait_min
func.func @gather_receiver()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 1, block_count = 6} {dfb_id = 1 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 6>
  ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 0 name "gather" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 2, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 3, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 4, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_dst):
    %recv = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 6>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf = ttl.copy %pipe, %recv
        : (!ttl.selected_pipe_dst,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf : !ttl.transfer_handle
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 6>
    ttl.yield
  }
  func.return
}

// CHECK-LABEL: func.func @gather_senders
// CHECK: scf.for %[[INDEX:.*]] =
// CHECK: %[[READY_INDEX:.*]] = ttkernel.experimental.constant_table_lookup %[[INDEX]], [6, 6, 6, 6, 6, 7] : index
// CHECK: %[[READY_SEM:.*]] = ttkernel.get_semaphore(%[[READY_INDEX]])
// CHECK: %[[READY_ADDR:.*]] = ttkernel.reinterpret_cast(%[[READY_SEM]])
// CHECK: %[[COMPLETION_INDEX:.*]] = ttkernel.experimental.constant_table_lookup %[[INDEX]], [0, 2, 3, 4, 5, 1] : index
// CHECK: %[[COMPLETION_SEM:.*]] = ttkernel.get_semaphore(%[[COMPLETION_INDEX]])
// CHECK: %[[COMPLETION_NOC_ADDR:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[COMPLETION_SEM]], {{.*}})
// CHECK: ttkernel.experimental.semaphore_wait(%[[READY_ADDR]], {{.*}})
// CHECK: ttkernel.noc_async_write %
// CHECK: ttkernel.noc_semaphore_inc(%[[COMPLETION_NOC_ADDR]], {{.*}})
func.func @gather_senders()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 0 name "gather" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 2, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 3, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 4, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    %xf = ttl.copy %cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.selected_pipe_src)
        -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
    ttl.yield
  }
  func.return
}
}
