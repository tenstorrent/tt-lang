// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=false})' | FileCheck %s

// Summary: Verifies shared local and GlobalSemaphore allocation for PipeNet
// completion and sender-ready counters.

// Sixteen transfers sharing a receiver consume all local semaphore ids. The
// sender-ready counter uses one GlobalSemaphore address whose per-core
// instances provide independent state on the two source cores.
// CHECK-LABEL: module attributes
// CHECK-SAME: ttl.pipe_global_semaphore_count = 1 : i64
// CHECK-SAME: ttl.pipe_sram_scratch_bytes = 32 : i64
// CHECK-SAME: ttl.pipe_sync_semaphore_count = 16 : i64
// CHECK-LABEL: func.func @completion_limit_uses_global_ready_counters
// CHECK-DAG: %[[SEM0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[SEM1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[SEM2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[SEM3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[SEM4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[SEM5:.*]] = arith.constant 5 : index
// CHECK-DAG: %[[SEM6:.*]] = arith.constant 6 : index
// CHECK-DAG: %[[SEM7:.*]] = arith.constant 7 : index
// CHECK-DAG: %[[SEM8:.*]] = arith.constant 8 : index
// CHECK-DAG: %[[SEM9:.*]] = arith.constant 9 : index
// CHECK-DAG: %[[SEM10:.*]] = arith.constant 10 : index
// CHECK-DAG: %[[SEM11:.*]] = arith.constant 11 : index
// CHECK-DAG: %[[SEM12:.*]] = arith.constant 12 : index
// CHECK-DAG: %[[SEM13:.*]] = arith.constant 13 : index
// CHECK-DAG: %[[SEM14:.*]] = arith.constant 14 : index
// CHECK-DAG: %[[SEM15:.*]] = arith.constant 15 : index
// CHECK: %[[SRC0_READY_POST:.*]] = ttkernel.get_common_arg_val(%[[SEM1]])
// CHECK: %[[SRC0_READY_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[SRC0_READY_POST]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc(%[[SRC0_READY_NOC]]
// CHECK: %[[SRC0_READY_SEND:.*]] = ttkernel.get_common_arg_val(%[[SEM1]])
// CHECK: %[[SRC0_READY_PTR:.*]] = ttkernel.reinterpret_cast(%[[SRC0_READY_SEND]])
// CHECK: ttkernel.experimental.semaphore_wait(%[[SRC0_READY_PTR]]
// CHECK: ttkernel.noc_semaphore_set(%[[SRC0_READY_PTR]]
// CHECK: %[[DONE0:.*]] = ttkernel.get_semaphore(%[[SEM0]])
// CHECK: %[[DONE0_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[DONE0]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc(%[[DONE0_NOC]]
// CHECK-NOT: ttkernel.get_semaphore
// CHECK: %[[SRC1_READY_POST:.*]] = ttkernel.get_common_arg_val(%[[SEM1]])
// CHECK: %[[SRC1_READY_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[SRC1_READY_POST]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc(%[[SRC1_READY_NOC]]
// CHECK: %[[SRC1_READY_SEND:.*]] = ttkernel.get_common_arg_val(%[[SEM1]])
// CHECK: %[[SRC1_READY_PTR:.*]] = ttkernel.reinterpret_cast(%[[SRC1_READY_SEND]])
// CHECK: ttkernel.experimental.semaphore_wait(%[[SRC1_READY_PTR]]
// CHECK: ttkernel.noc_semaphore_set(%[[SRC1_READY_PTR]]
// CHECK: %[[DONE1:.*]] = ttkernel.get_semaphore(%[[SEM1]])
// CHECK: %[[DONE1_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[DONE1]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc(%[[DONE1_NOC]]
// CHECK-NOT: ttkernel.get_semaphore
// CHECK: ttkernel.get_semaphore(%[[SEM2]])
// CHECK-NOT: ttkernel.get_semaphore
// CHECK: ttkernel.get_semaphore(%[[SEM3]])
// CHECK-NOT: ttkernel.get_semaphore
// CHECK: ttkernel.get_semaphore(%[[SEM4]])
// CHECK-NOT: ttkernel.get_semaphore
// CHECK: ttkernel.get_semaphore(%[[SEM5]])
// CHECK-NOT: ttkernel.get_semaphore
// CHECK: ttkernel.get_semaphore(%[[SEM6]])
// CHECK-NOT: ttkernel.get_semaphore
// CHECK: ttkernel.get_semaphore(%[[SEM7]])
// CHECK-NOT: ttkernel.get_semaphore
// CHECK: ttkernel.get_semaphore(%[[SEM8]])
// CHECK-NOT: ttkernel.get_semaphore
// CHECK: ttkernel.get_semaphore(%[[SEM9]])
// CHECK-NOT: ttkernel.get_semaphore
// CHECK: ttkernel.get_semaphore(%[[SEM10]])
// CHECK-NOT: ttkernel.get_semaphore
// CHECK: ttkernel.get_semaphore(%[[SEM11]])
// CHECK-NOT: ttkernel.get_semaphore
// CHECK: ttkernel.get_semaphore(%[[SEM12]])
// CHECK-NOT: ttkernel.get_semaphore
// CHECK: ttkernel.get_semaphore(%[[SEM13]])
// CHECK-NOT: ttkernel.get_semaphore
// CHECK: ttkernel.get_semaphore(%[[SEM14]])
// CHECK-NOT: ttkernel.get_semaphore
// CHECK: ttkernel.get_semaphore(%[[SEM15]])
// CHECK-NOT: ttkernel.get_semaphore
// CHECK: return
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @completion_limit_uses_global_ready_counters()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst0 = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst1 = ttl.bind_cb {cb_index = 2, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst2 = ttl.bind_cb {cb_index = 3, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst3 = ttl.bind_cb {cb_index = 4, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst4 = ttl.bind_cb {cb_index = 5, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst5 = ttl.bind_cb {cb_index = 6, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst6 = ttl.bind_cb {cb_index = 7, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst7 = ttl.bind_cb {cb_index = 8, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst8 = ttl.bind_cb {cb_index = 9, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst9 = ttl.bind_cb {cb_index = 10, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst10 = ttl.bind_cb {cb_index = 11, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst11 = ttl.bind_cb {cb_index = 12, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst12 = ttl.bind_cb {cb_index = 13, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst13 = ttl.bind_cb {cb_index = 14, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst14 = ttl.bind_cb {cb_index = 15, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst15 = ttl.bind_cb {cb_index = 16, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %p0 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
    %p1 = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 1
        : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>
    %p2 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 2
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 2>
    %p3 = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 3
        : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 3>
    %p4 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 4
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 4>
    %p5 = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 5
        : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 5>
    %p6 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 6
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 6>
    %p7 = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 7
        : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 7>
    %p8 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 8
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 8>
    %p9 = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 9
        : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 9>
    %p10 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 10
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 10>
    %p11 = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 11
        : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 11>
    %p12 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 12
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 12>
    %p13 = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 13
        : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 13>
    %p14 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 14
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 14>
    %p15 = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 15
        : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 15>
    %recv0 = ttl.cb_reserve %dst0
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post0 = ttl.copy %p0, %recv0
        : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send0 = ttl.copy %src, %p0
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>)
        -> !ttl.transfer_handle<write>
    %recv1 = ttl.cb_reserve %dst1
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post1 = ttl.copy %p1, %recv1
        : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send1 = ttl.copy %src, %p1
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>)
        -> !ttl.transfer_handle<write>
    %recv2 = ttl.cb_reserve %dst2
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post2 = ttl.copy %p2, %recv2
        : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 2>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send2 = ttl.copy %src, %p2
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 2>)
        -> !ttl.transfer_handle<write>
    %recv3 = ttl.cb_reserve %dst3
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post3 = ttl.copy %p3, %recv3
        : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 3>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send3 = ttl.copy %src, %p3
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 3>)
        -> !ttl.transfer_handle<write>
    %recv4 = ttl.cb_reserve %dst4
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post4 = ttl.copy %p4, %recv4
        : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 4>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send4 = ttl.copy %src, %p4
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 4>)
        -> !ttl.transfer_handle<write>
    %recv5 = ttl.cb_reserve %dst5
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post5 = ttl.copy %p5, %recv5
        : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 5>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send5 = ttl.copy %src, %p5
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 5>)
        -> !ttl.transfer_handle<write>
    %recv6 = ttl.cb_reserve %dst6
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post6 = ttl.copy %p6, %recv6
        : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 6>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send6 = ttl.copy %src, %p6
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 6>)
        -> !ttl.transfer_handle<write>
    %recv7 = ttl.cb_reserve %dst7
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post7 = ttl.copy %p7, %recv7
        : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 7>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send7 = ttl.copy %src, %p7
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 7>)
        -> !ttl.transfer_handle<write>
    %recv8 = ttl.cb_reserve %dst8
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post8 = ttl.copy %p8, %recv8
        : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 8>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send8 = ttl.copy %src, %p8
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 8>)
        -> !ttl.transfer_handle<write>
    %recv9 = ttl.cb_reserve %dst9
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post9 = ttl.copy %p9, %recv9
        : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 9>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send9 = ttl.copy %src, %p9
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 9>)
        -> !ttl.transfer_handle<write>
    %recv10 = ttl.cb_reserve %dst10
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post10 = ttl.copy %p10, %recv10
        : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 10>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send10 = ttl.copy %src, %p10
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 10>)
        -> !ttl.transfer_handle<write>
    %recv11 = ttl.cb_reserve %dst11
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post11 = ttl.copy %p11, %recv11
        : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 11>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send11 = ttl.copy %src, %p11
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 11>)
        -> !ttl.transfer_handle<write>
    %recv12 = ttl.cb_reserve %dst12
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post12 = ttl.copy %p12, %recv12
        : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 12>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send12 = ttl.copy %src, %p12
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 12>)
        -> !ttl.transfer_handle<write>
    %recv13 = ttl.cb_reserve %dst13
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post13 = ttl.copy %p13, %recv13
        : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 13>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send13 = ttl.copy %src, %p13
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 13>)
        -> !ttl.transfer_handle<write>
    %recv14 = ttl.cb_reserve %dst14
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post14 = ttl.copy %p14, %recv14
        : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 14>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send14 = ttl.copy %src, %p14
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 14>)
        -> !ttl.transfer_handle<write>
    %recv15 = ttl.cb_reserve %dst15
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post15 = ttl.copy %p15, %recv15
        : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 15>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send15 = ttl.copy %src, %p15
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 15>)
        -> !ttl.transfer_handle<write>
    func.return
  }
}
// -----

// Seventeen transfers sharing a receiver use all sixteen local semaphore ids.
// One completion counter and two overlapping sender-ready counters use
// GlobalSemaphore storage.
// CHECK-LABEL: module attributes
// CHECK-SAME: ttl.pipe_global_semaphore_count = 3 : i64
// CHECK-SAME: ttl.pipe_sram_scratch_bytes = 32 : i64
// CHECK-SAME: ttl.pipe_sync_semaphore_count = 16 : i64
// CHECK-LABEL: func.func @completion_overflow_uses_global_counter
// CHECK-DAG: %[[GLOBAL_COMPLETION_INDEX:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[READY0_INDEX:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[READY1_INDEX:.*]] = arith.constant 3 : index
// CHECK: %[[READY0_POST:.*]] = ttkernel.get_common_arg_val(%[[READY0_INDEX]])
// CHECK: %[[READY0_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[READY0_POST]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc(%[[READY0_NOC]]
// CHECK: %[[READY0_SEND:.*]] = ttkernel.get_common_arg_val(%[[READY0_INDEX]])
// CHECK: %[[READY0_PTR:.*]] = ttkernel.reinterpret_cast(%[[READY0_SEND]])
// CHECK: ttkernel.experimental.semaphore_wait(%[[READY0_PTR]]
// CHECK: ttkernel.noc_semaphore_set(%[[READY0_PTR]]
// CHECK: %[[READY1_POST:.*]] = ttkernel.get_common_arg_val(%[[READY1_INDEX]])
// CHECK: %[[READY1_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[READY1_POST]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc(%[[READY1_NOC]]
// CHECK: %[[GLOBAL_COMPLETION_WAIT:.*]] = ttkernel.get_common_arg_val(%[[GLOBAL_COMPLETION_INDEX]])
// CHECK: %[[GLOBAL_COMPLETION_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[GLOBAL_COMPLETION_WAIT]])
// CHECK: ttkernel.experimental.semaphore_wait_min(%[[GLOBAL_COMPLETION_PTR]]
// CHECK: ttkernel.cb_pop_front
// CHECK: %[[READY1_SEND:.*]] = ttkernel.get_common_arg_val(%[[READY1_INDEX]])
// CHECK: %[[READY1_PTR:.*]] = ttkernel.reinterpret_cast(%[[READY1_SEND]])
// CHECK: ttkernel.experimental.semaphore_wait(%[[READY1_PTR]]
// CHECK: ttkernel.noc_semaphore_set(%[[READY1_PTR]]
// CHECK: ttkernel.noc_async_write
// CHECK: ttkernel.noc_async_write_barrier
// CHECK: %[[GLOBAL_COMPLETION_SEND:.*]] = ttkernel.get_common_arg_val(%[[GLOBAL_COMPLETION_INDEX]])
// CHECK: %[[GLOBAL_COMPLETION_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[GLOBAL_COMPLETION_SEND]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc(%[[GLOBAL_COMPLETION_NOC]]
// CHECK: return
module attributes {ttl.launch_grid = array<i64: 17, 1>} {
  func.func @completion_overflow_uses_global_counter()
      attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %p = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 16
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 16>
    %dst0 = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst1 = ttl.bind_cb {cb_index = 2, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst2 = ttl.bind_cb {cb_index = 3, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst3 = ttl.bind_cb {cb_index = 4, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst4 = ttl.bind_cb {cb_index = 5, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst5 = ttl.bind_cb {cb_index = 6, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst6 = ttl.bind_cb {cb_index = 7, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst7 = ttl.bind_cb {cb_index = 8, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst8 = ttl.bind_cb {cb_index = 9, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst9 = ttl.bind_cb {cb_index = 10, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst10 = ttl.bind_cb {cb_index = 11, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst11 = ttl.bind_cb {cb_index = 12, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst12 = ttl.bind_cb {cb_index = 13, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst13 = ttl.bind_cb {cb_index = 14, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst14 = ttl.bind_cb {cb_index = 15, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst15 = ttl.bind_cb {cb_index = 16, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %p0 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 0
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 0>
    %p1 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 1
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 1>
    %p2 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 2
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 2>
    %p3 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 3
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 3>
    %p4 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 4
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 4>
    %p5 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 5
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 5>
    %p6 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 6
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 6>
    %p7 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 7
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 7>
    %p8 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 8
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 8>
    %p9 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 9
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 9>
    %p10 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 10
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 10>
    %p11 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 11
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 11>
    %p12 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 12
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 12>
    %p13 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 13
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 13>
    %p14 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 14
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 14>
    %p15 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 15
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 15>
    %transfer15 = ttl.pipe_transfer.create %p15 {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 15>
        -> !ttl.pipe_transfer
    %recv = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post = ttl.copy %p, %recv
        : (!ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 16>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send = ttl.copy %cb, %p
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 16>)
        -> !ttl.transfer_handle<write>
    %recv0 = ttl.cb_reserve %dst0
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post0 = ttl.copy %p0, %recv0
        : (!ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send0 = ttl.copy %cb, %p0
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 0>)
        -> !ttl.transfer_handle<write>
    %recv1 = ttl.cb_reserve %dst1
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post1 = ttl.copy %p1, %recv1
        : (!ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 1>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send1 = ttl.copy %cb, %p1
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 1>)
        -> !ttl.transfer_handle<write>
    %recv2 = ttl.cb_reserve %dst2
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post2 = ttl.copy %p2, %recv2
        : (!ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 2>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send2 = ttl.copy %cb, %p2
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 2>)
        -> !ttl.transfer_handle<write>
    %recv3 = ttl.cb_reserve %dst3
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post3 = ttl.copy %p3, %recv3
        : (!ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 3>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send3 = ttl.copy %cb, %p3
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 3>)
        -> !ttl.transfer_handle<write>
    %recv4 = ttl.cb_reserve %dst4
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post4 = ttl.copy %p4, %recv4
        : (!ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 4>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send4 = ttl.copy %cb, %p4
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 4>)
        -> !ttl.transfer_handle<write>
    %recv5 = ttl.cb_reserve %dst5
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post5 = ttl.copy %p5, %recv5
        : (!ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 5>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send5 = ttl.copy %cb, %p5
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 5>)
        -> !ttl.transfer_handle<write>
    %recv6 = ttl.cb_reserve %dst6
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post6 = ttl.copy %p6, %recv6
        : (!ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 6>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send6 = ttl.copy %cb, %p6
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 6>)
        -> !ttl.transfer_handle<write>
    %recv7 = ttl.cb_reserve %dst7
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post7 = ttl.copy %p7, %recv7
        : (!ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 7>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send7 = ttl.copy %cb, %p7
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 7>)
        -> !ttl.transfer_handle<write>
    %recv8 = ttl.cb_reserve %dst8
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post8 = ttl.copy %p8, %recv8
        : (!ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 8>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send8 = ttl.copy %cb, %p8
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 8>)
        -> !ttl.transfer_handle<write>
    %recv9 = ttl.cb_reserve %dst9
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post9 = ttl.copy %p9, %recv9
        : (!ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 9>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send9 = ttl.copy %cb, %p9
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 9>)
        -> !ttl.transfer_handle<write>
    %recv10 = ttl.cb_reserve %dst10
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post10 = ttl.copy %p10, %recv10
        : (!ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 10>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send10 = ttl.copy %cb, %p10
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 10>)
        -> !ttl.transfer_handle<write>
    %recv11 = ttl.cb_reserve %dst11
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post11 = ttl.copy %p11, %recv11
        : (!ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 11>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send11 = ttl.copy %cb, %p11
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 11>)
        -> !ttl.transfer_handle<write>
    %recv12 = ttl.cb_reserve %dst12
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post12 = ttl.copy %p12, %recv12
        : (!ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 12>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send12 = ttl.copy %cb, %p12
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 12>)
        -> !ttl.transfer_handle<write>
    %recv13 = ttl.cb_reserve %dst13
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post13 = ttl.copy %p13, %recv13
        : (!ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 13>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send13 = ttl.copy %cb, %p13
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 13>)
        -> !ttl.transfer_handle<write>
    %recv14 = ttl.cb_reserve %dst14
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post14 = ttl.copy %p14, %recv14
        : (!ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 14>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    %send14 = ttl.copy %cb, %p14
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 14>)
        -> !ttl.transfer_handle<write>
    ttl.if_dst %p15
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 15> {
      %recv15 = ttl.cb_reserve %dst15
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post15 = ttl.pipe_transfer.post %transfer15, %recv15
          : (!ttl.pipe_transfer,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.pipe_token<net 15>
      ttl.pipe_transfer.wait %post15 : !ttl.pipe_token<net 15>
      ttl.cb_push %dst15 : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      %ready15 = ttl.cb_wait %dst15
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst15 : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    ttl.if_src %p15
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 15> {
      %send15 = ttl.pipe_transfer.send %transfer15, %cb
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send15 : !ttl.transfer_handle<write>
    }
    func.return
  }
}
