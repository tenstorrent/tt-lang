// Verifies rotating wait-any lowering for local and GlobalSemaphore-backed
// PipeNet completion counters.

// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-global-semaphores-only=false})' | FileCheck %s --check-prefix=LOCAL
// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-global-semaphores-only=true})' | FileCheck %s --check-prefix=GLOBAL

// LOCAL-LABEL: func.func @ready_receive
// LOCAL: %[[COUNT:.*]] = arith.constant 2 : i32
// LOCAL: %[[START_I32:.*]] = arith.index_cast %{{.*}} : index to i32
// LOCAL: %[[REMAINDER:.*]] = arith.remsi %[[START_I32]], %[[COUNT]] : i32
// LOCAL: %[[NONNEGATIVE:.*]] = arith.addi %[[REMAINDER]], %[[COUNT]] : i32
// LOCAL: %[[START:.*]] = arith.remui %[[NONNEGATIVE]], %[[COUNT]] : i32
// LOCAL: %[[SELECTED:.*]] = scf.while
// LOCAL: scf.for
// LOCAL: scf.index_switch
// LOCAL-COUNT-2: ttkernel.experimental.semaphore_reached
// LOCAL: arith.select
// LOCAL: arith.index_cast %[[SELECTED]] : i32 to index
// LOCAL-COUNT-2: ttkernel.experimental.semaphore_wait_min
// LOCAL-NOT: ttl.wait_any
// LOCAL-NOT: ttl.pipe_transfer.wait_any

// GLOBAL-LABEL: func.func @ready_receive
// GLOBAL: ttkernel.get_common_arg_val
// GLOBAL: scf.while
// GLOBAL: scf.for
// GLOBAL: scf.index_switch
// GLOBAL-COUNT-2: ttkernel.experimental.semaphore_reached
// GLOBAL-NOT: ttkernel.get_semaphore

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @ready_receive(%start: index) -> index
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %landing0 = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %landing1 = ttl.bind_cb {cb_index = 2, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
    %dst0 = ttl.cb_reserve %landing0
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %dst1 = ttl.cb_reserve %landing1
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %request0 = ttl.copy %pipe0, %dst0
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.receive_request
    %request1 = ttl.copy %pipe1, %dst1
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.receive_request
    %send0 = ttl.copy %source, %pipe0
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send0 : !ttl.transfer_handle<write>
    %send1 = ttl.copy %source, %pipe1
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send1 : !ttl.transfer_handle<write>
    %ready = ttl.wait_any %request0, %request1 start %start
        : (!ttl.receive_request, !ttl.receive_request, index)
        -> !ttl.ready_receive
    %selected = ttl.ready_receive_index %ready : !ttl.ready_receive
    ttl.wait %request0 : !ttl.receive_request
    ttl.wait %request1 : !ttl.receive_request
    ttl.cb_push %landing0
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    ttl.cb_push %landing1
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    func.return %selected : index
  }
}

// -----

// A request merged from mutually exclusive posts is valid when every origin
// uses the same logical channel and destination DFB stream.
// LOCAL-LABEL: func.func @merged_request_origins
// LOCAL: scf.if
// LOCAL: scf.while
// LOCAL: ttkernel.experimental.semaphore_reached
// LOCAL: scf.if
// LOCAL: ttkernel.cb_push_back
// GLOBAL-LABEL: func.func @merged_request_origins
// GLOBAL: scf.if
// GLOBAL: scf.while
// GLOBAL: ttkernel.experimental.semaphore_reached

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @merged_request_origins(%condition: i1) -> index
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %landing = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %request = scf.if %condition -> (!ttl.receive_request) {
      %dst = ttl.cb_reserve %landing
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %then_request = ttl.copy %pipe, %dst
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      %send = ttl.copy %source, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      scf.yield %then_request : !ttl.receive_request
    } else {
      %dst = ttl.cb_reserve %landing
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %else_request = ttl.copy %pipe, %dst
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      %send = ttl.copy %source, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      scf.yield %else_request : !ttl.receive_request
    }
    %start = arith.constant 0 : index
    %ready = ttl.wait_any %request start %start
        : (!ttl.receive_request, index) -> !ttl.ready_receive
    %selected = ttl.ready_receive_index %ready : !ttl.ready_receive
    %zero = arith.constant 0 : index
    %is_not_selected = arith.cmpi ne, %selected, %zero : index
    scf.if %is_not_selected {
    } else {
      ttl.cb_push %landing
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    func.return %selected : index
  }
}
