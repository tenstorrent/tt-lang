// Verifies rotating wait-any lowering for local and GlobalSemaphore-backed
// PipeNet completion counters.

// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=true pipe-global-semaphores-only=false})' | FileCheck %s --check-prefix=LOCAL
// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-global-semaphores-only=true})' | FileCheck %s --check-prefix=GLOBAL
// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=true pipe-global-semaphores-only=false})' | FileCheck %s --check-prefix=SEMANTICS
// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=false pipe-global-semaphores-only=false})' | FileCheck %s --check-prefix=RECEIVER-POST
// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=false pipe-capacity-sync=false pipe-global-semaphores-only=false})' | FileCheck %s --check-prefix=PUBLISHED

// Verifies candidate-specific completion checks and ascending cyclic selection.
// SEMANTICS-LABEL: func.func @ready_receive
// SEMANTICS-DAG: %[[C0:.*]] = arith.constant 0 : index
// SEMANTICS-DAG: %[[C1:.*]] = arith.constant 1 : index
// SEMANTICS-DAG: %[[C2:.*]] = arith.constant 2 : index
// SEMANTICS-DAG: %[[C3:.*]] = arith.constant 3 : index
// SEMANTICS-DAG: %[[COUNT:.*]] = arith.constant 4 : i32
// SEMANTICS: %[[CTR0:.*]] = memref.alloca()
// SEMANTICS: %[[CTR1:.*]] = memref.alloca()
// SEMANTICS: %[[CTR2:.*]] = memref.alloca()
// SEMANTICS: %[[CTR3:.*]] = memref.alloca()
// SEMANTICS: %[[SEQ0:.*]] = arith.addi
// SEMANTICS: memref.store %[[SEQ0]], %[[CTR0]]
// SEMANTICS: %[[SEQ1:.*]] = arith.addi
// SEMANTICS: memref.store %[[SEQ1]], %[[CTR1]]
// SEMANTICS: %[[SEQ2:.*]] = arith.addi
// SEMANTICS: memref.store %[[SEQ2]], %[[CTR2]]
// SEMANTICS: %[[SEQ3:.*]] = arith.addi
// SEMANTICS: memref.store %[[SEQ3]], %[[CTR3]]
// SEMANTICS: %[[START:.*]] = arith.remui %{{.*}}, %[[COUNT]] : i32
// SEMANTICS: scf.while
// SEMANTICS: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step
// SEMANTICS:   %[[OFF:.*]] = arith.index_cast %[[IV]] : index to i32
// SEMANTICS:   %[[ROT:.*]] = arith.addi %[[START]], %[[OFF]] : i32
// SEMANTICS:   %[[CAND:.*]] = arith.remui %[[ROT]], %[[COUNT]] : i32
// SEMANTICS:   %[[CANDIDX:.*]] = arith.index_cast %[[CAND]] : i32 to index
// SEMANTICS:   scf.index_switch %[[CANDIDX]]
// SEMANTICS:   case 0 {
// SEMANTICS:     %[[SEM0:.*]] = ttkernel.get_semaphore(%[[C0]])
// SEMANTICS:     %[[PTR0:.*]] = ttkernel.reinterpret_cast(%[[SEM0]])
// SEMANTICS:     ttkernel.experimental.semaphore_reached(%[[PTR0]], %[[SEQ0]])
// SEMANTICS:   case 1 {
// SEMANTICS:     %[[SEM1:.*]] = ttkernel.get_semaphore(%[[C1]])
// SEMANTICS:     %[[PTR1:.*]] = ttkernel.reinterpret_cast(%[[SEM1]])
// SEMANTICS:     ttkernel.experimental.semaphore_reached(%[[PTR1]], %[[SEQ1]])
// SEMANTICS:   case 2 {
// SEMANTICS:     %[[SEM2:.*]] = ttkernel.get_semaphore(%[[C2]])
// SEMANTICS:     %[[PTR2:.*]] = ttkernel.reinterpret_cast(%[[SEM2]])
// SEMANTICS:     ttkernel.experimental.semaphore_reached(%[[PTR2]], %[[SEQ2]])
// SEMANTICS:   case 3 {
// SEMANTICS:     %[[SEM3:.*]] = ttkernel.get_semaphore(%[[C3]])
// SEMANTICS:     %[[PTR3:.*]] = ttkernel.reinterpret_cast(%[[SEM3]])
// SEMANTICS:     ttkernel.experimental.semaphore_reached(%[[PTR3]], %[[SEQ3]])
// SEMANTICS:   %[[NEXT:.*]] = arith.select %{{.*}}, %[[CAND]], %{{.*}} : i32

// LOCAL-LABEL: func.func @ready_receive
// LOCAL-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1, 2, 3, 4>
// LOCAL: %[[COUNT:.*]] = arith.constant 4 : i32
// LOCAL-COUNT-4: ttkernel.noc_async_write_multicast_loopback_src
// LOCAL: %[[START_I32:.*]] = arith.index_cast %{{.*}} : index to i32
// LOCAL: %[[REMAINDER:.*]] = arith.remsi %[[START_I32]], %[[COUNT]] : i32
// LOCAL: %[[NONNEGATIVE:.*]] = arith.addi %[[REMAINDER]], %[[COUNT]] : i32
// LOCAL: %[[START:.*]] = arith.remui %[[NONNEGATIVE]], %[[COUNT]] : i32
// LOCAL: %[[SELECTED:.*]] = scf.while
// LOCAL: scf.for
// LOCAL: scf.index_switch
// LOCAL-COUNT-4: ttkernel.experimental.semaphore_reached
// LOCAL: arith.select
// LOCAL: arith.index_cast %[[SELECTED]] : i32 to index
// LOCAL-COUNT-4: ttkernel.experimental.semaphore_wait_min
// LOCAL-NOT: ttl.wait_any
// LOCAL-NOT: ttl.pipe_transfer.wait_any

// GLOBAL-LABEL: func.func @ready_receive
// GLOBAL: ttkernel.get_common_arg_val
// GLOBAL: scf.while
// GLOBAL: scf.for
// GLOBAL: scf.index_switch
// GLOBAL-COUNT-4: ttkernel.experimental.semaphore_reached
// GLOBAL-NOT: ttkernel.get_semaphore

// RECEIVER-POST-LABEL: func.func @ready_receive
// RECEIVER-POST-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1, 2, 3, 4>
// RECEIVER-POST-NOT: ttkernel.store_to_l1
// RECEIVER-POST: scf.while
// RECEIVER-POST-COUNT-4: ttkernel.experimental.semaphore_reached
// RECEIVER-POST-NOT: ttkernel.load_from_l1
// RECEIVER-POST: return

// PUBLISHED-LABEL: func.func @ready_receive
// PUBLISHED-NOT: ttl.pipe_computed_address_dfb_indices
// PUBLISHED-COUNT-4: ttkernel.noc_inline_dw_write
// PUBLISHED-COUNT-4: ttkernel.noc_async_write_multicast_loopback_src
// PUBLISHED: scf.while
// PUBLISHED-COUNT-4: ttkernel.experimental.semaphore_reached

// Four multicast candidates on a two-node grid cover candidate-resource
// mapping independently of the single-candidate control-flow case below.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @ready_receive(%start: index) -> index
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %landing0 = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %landing1 = ttl.bind_cb {cb_index = 2, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %landing2 = ttl.bind_cb {cb_index = 3, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %landing3 = ttl.bind_cb {cb_index = 4, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>
    %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(1, 0) net 1
        : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 1>
    %pipe2 = ttl.create_pipe src(0, 0) dst(0, 0) to(1, 0) net 2
        : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 2>
    %pipe3 = ttl.create_pipe src(0, 0) dst(0, 0) to(1, 0) net 3
        : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 3>
    %dst0 = ttl.cb_reserve %landing0
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %dst1 = ttl.cb_reserve %landing1
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %dst2 = ttl.cb_reserve %landing2
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %dst3 = ttl.cb_reserve %landing3
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %request0 = ttl.copy %pipe0, %dst0
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.receive_request
    %request1 = ttl.copy %pipe1, %dst1
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 1>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.receive_request
    %request2 = ttl.copy %pipe2, %dst2
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 2>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.receive_request
    %request3 = ttl.copy %pipe3, %dst3
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 3>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.receive_request
    %send0 = ttl.copy %source, %pipe0
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send0 : !ttl.transfer_handle<write>
    %send1 = ttl.copy %source, %pipe1
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 1>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send1 : !ttl.transfer_handle<write>
    %send2 = ttl.copy %source, %pipe2
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 2>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send2 : !ttl.transfer_handle<write>
    %send3 = ttl.copy %source, %pipe3
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 3>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send3 : !ttl.transfer_handle<write>
    %ready = ttl.wait_any %request0, %request1, %request2, %request3 start %start
        : (!ttl.receive_request, !ttl.receive_request, !ttl.receive_request,
           !ttl.receive_request, index)
        -> !ttl.ready_receive
    %selected = ttl.ready_receive_index %ready : !ttl.ready_receive
    ttl.wait %request0 : !ttl.receive_request
    ttl.wait %request1 : !ttl.receive_request
    ttl.wait %request2 : !ttl.receive_request
    ttl.wait %request3 : !ttl.receive_request
    ttl.cb_push %landing0
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    ttl.cb_push %landing1
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    ttl.cb_push %landing2
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    ttl.cb_push %landing3
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    func.return %selected : index
  }
}

// -----

// Capacity-eligible point-to-point candidates verify that wait-any preserves
// sender capacity acquisition and receiver capacity release.
// LOCAL-LABEL: func.func @ready_receive_capacity
// LOCAL-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1, 2>
// LOCAL-DAG: %[[CAPACITY0:.*]] = arith.constant 2 : index
// LOCAL-DAG: %[[CAPACITY1:.*]] = arith.constant 3 : index
// LOCAL: %[[CAPACITY0_INIT_SEM:.*]] = ttkernel.get_semaphore(%[[CAPACITY0]])
// LOCAL-NEXT: %[[CAPACITY0_INIT:.*]] = ttkernel.reinterpret_cast(%[[CAPACITY0_INIT_SEM]])
// LOCAL-NEXT: ttkernel.noc_semaphore_set(%[[CAPACITY0_INIT]]
// LOCAL: %[[CAPACITY1_INIT_SEM:.*]] = ttkernel.get_semaphore(%[[CAPACITY1]])
// LOCAL-NEXT: %[[CAPACITY1_INIT:.*]] = ttkernel.reinterpret_cast(%[[CAPACITY1_INIT_SEM]])
// LOCAL-NEXT: ttkernel.noc_semaphore_set(%[[CAPACITY1_INIT]]
// LOCAL: %[[CAPACITY0_ACQUIRE_SEM:.*]] = ttkernel.get_semaphore(%[[CAPACITY0]])
// LOCAL-NEXT: %[[CAPACITY0_ACQUIRE:.*]] = ttkernel.reinterpret_cast(%[[CAPACITY0_ACQUIRE_SEM]])
// LOCAL: ttkernel.experimental.semaphore_wait_min(%[[CAPACITY0_ACQUIRE]]
// LOCAL: ttkernel.noc_async_write
// LOCAL: %[[CAPACITY1_ACQUIRE_SEM:.*]] = ttkernel.get_semaphore(%[[CAPACITY1]])
// LOCAL-NEXT: %[[CAPACITY1_ACQUIRE:.*]] = ttkernel.reinterpret_cast(%[[CAPACITY1_ACQUIRE_SEM]])
// LOCAL: ttkernel.experimental.semaphore_wait_min(%[[CAPACITY1_ACQUIRE]]
// LOCAL: ttkernel.noc_async_write
// LOCAL: scf.while
// LOCAL: ttkernel.cb_pop_front
// LOCAL: ttkernel.noc_semaphore_inc
// LOCAL: ttkernel.cb_pop_front
// LOCAL: ttkernel.noc_semaphore_inc
// LOCAL-NOT: ttkernel.noc_inline_dw_write
// LOCAL: return

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @ready_receive_capacity(%start: index) -> index
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %landing0 = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %landing1 = ttl.bind_cb {cb_index = 2, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
    %dst0 = ttl.cb_reserve %landing0
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %dst1 = ttl.cb_reserve %landing1
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
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
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.cb_push %landing1
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    %block0 = ttl.cb_wait %landing0
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_pop %landing0
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    %block1 = ttl.cb_wait %landing1
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_pop %landing1
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
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
