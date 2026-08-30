// Summary: Pipe transport planning records backend-independent scalar
// schedules, endpoint storage, receiver address recurrences, and completion
// groups.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=false})' -debug-only=ttl-pipe-transport-plan 2>&1 >/dev/null | FileCheck %s --check-prefix=PLAN
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=false})' | FileCheck %s --check-prefix=LOWERED

// Purpose: a point-to-point stream records one logical transfer per group and
// releases source storage after its only endpoint completes.
// PLAN: PipeTransport: stream 0 transfer 0 src(0, 0) -> dst(1, 0) to (1, 0) net 0 contract=point_to_point synchronization=receiver_post schedule=scalar credit_completion=immediate group=1
// PLAN-NEXT: PipeTransport:   source blocks=2 block_span=1 stage_depth=2 ownership=dfb scratch_offset=0 scratch_bytes=0 pages=1 page_bytes=4096 loops=0
// PLAN-NEXT: PipeTransport:   endpoint 0 dst(1, 0) DFB 1 block_count=2 slot_span=1 group_depth=1 ownership=dfb scratch_offset=0 scratch_bytes=0 loops=0 address=recurrence(initial=0, stride=1, modulus=2, executions=1)
// PLAN-NEXT: PipeTransport:   completion endpoints=[0] source_reuse=after_completion_group
// LOWERED-LABEL: func.func @point_to_point_transport
// LOWERED-NOT: ttkernel.noc_async_write_barrier
// LOWERED: ttkernel.noc_async_write_one_packet_set_state({{.*}}) posted true
// LOWERED: ttkernel.experimental.semaphore_wait(
// LOWERED: ttkernel.noc_semaphore_set
// LOWERED: ttkernel.noc_async_write_one_packet_with_state({{.*}}) posted true
// LOWERED-NEXT: ttkernel.noc_inline_dw_write({{.*}}) posted true
// LOWERED-NEXT: ttkernel.noc_async_writes_flushed({{.*}}) posted true
// LOWERED-NOT: ttkernel.noc_async_write_barrier
// LOWERED-NOT: ttkernel.noc_async_atomic_barrier
// LOWERED: return
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @point_to_point_transport()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      %ready = ttl.cb_wait %dst_dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst_dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.pipe_transfer.send %transfer, %src_dfb
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Purpose: a collective stream records one endpoint per receiver and requires
// every destination transfer to complete before source storage is reused.
// PLAN: PipeTransport: stream 0 transfer 0 src(0, 0) -> dst(1, 0) to (2, 0) net 0 contract=collective synchronization=receiver_post schedule=scalar credit_completion=immediate group=1
// PLAN-NEXT: PipeTransport:   source blocks=2 block_span=1 stage_depth=2 ownership=dfb scratch_offset=0 scratch_bytes=0 pages=1 page_bytes=4096 loops=0
// PLAN-NEXT: PipeTransport:   endpoint 0 dst(1, 0) DFB 5 block_count=2 slot_span=1 group_depth=1 ownership=dfb scratch_offset=0 scratch_bytes=0 loops=0 address=recurrence(initial=0, stride=1, modulus=2, executions=1)
// PLAN-NEXT: PipeTransport:   endpoint 1 dst(2, 0) DFB 5 block_count=2 slot_span=1 group_depth=1 ownership=dfb scratch_offset=0 scratch_bytes=0 loops=0 address=recurrence(initial=0, stride=1, modulus=2, executions=1)
// PLAN-NEXT: PipeTransport:   completion endpoints=[0, 1] source_reuse=after_completion_group
// LOWERED-LABEL: func.func @collective_transport
// LOWERED-NOT: posted true
// LOWERED: ttkernel.noc_async_write_multicast
// LOWERED-NEXT: ttkernel.noc_async_write_barrier
// LOWERED-NEXT: ttkernel.noc_semaphore_inc_multicast
// LOWERED-NEXT: ttkernel.noc_async_atomic_barrier
// LOWERED-NOT: posted true
// LOWERED: return
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @collective_transport()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_dfb = ttl.bind_cb {cb_index = 5, block_count = 2} {dfb_id = 5 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0 {
        isCollective = true}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {
        expectedReceivers = 2 : i64,
        kind = #ttl.pipe_transfer_kind<collective>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
        -> !ttl.pipe_transfer
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> {
      %recv = ttl.cb_reserve %dst_dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      %ready = ttl.cb_wait %dst_dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst_dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> {
      %send = ttl.pipe_transfer.send %transfer, %src_dfb
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Purpose: a grouped stream records the logical transfer count independently
// from the selected receiver group depth.
// PLAN: PipeTransport: stream 0 transfer 0 src(0, 0) -> dst(1, 0) to (1, 0) net 0 contract=point_to_point synchronization=receiver_post schedule=grouped credit_completion=immediate group=2
// PLAN-NEXT: PipeTransport:   source blocks=4 block_span=2 stage_depth=2 ownership=dfb scratch_offset=0 scratch_bytes=0 pages=2 page_bytes=4096 loops=0
// PLAN-NEXT: PipeTransport:   endpoint 0 dst(1, 0) DFB 1 block_count=4 slot_span=2 group_depth=2 ownership=dfb scratch_offset=0 scratch_bytes=0 loops=0 address=recurrence(initial=0, stride=2, modulus=4, executions=1)
// PLAN-NEXT: PipeTransport:   completion endpoints=[0] source_reuse=after_completion_group
// LOWERED-LABEL: func.func @grouped_transport
// LOWERED-NOT: ttkernel.noc_async_write_barrier
// LOWERED: ttkernel.noc_async_write_one_packet_set_state({{.*}}) posted true
// LOWERED: ttkernel.experimental.semaphore_wait(
// LOWERED: ttkernel.noc_semaphore_set
// LOWERED: ttkernel.noc_async_write_one_packet_with_state({{.*}}) posted true
// LOWERED-NEXT: ttkernel.noc_inline_dw_write({{.*}}) posted true
// LOWERED-NEXT: ttkernel.noc_async_writes_flushed({{.*}}) posted true
// LOWERED-NOT: ttkernel.noc_async_write_barrier
// LOWERED-NOT: ttkernel.noc_async_atomic_barrier
// LOWERED: return
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @grouped_transport()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 4} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>
    %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 4} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {
        block_span = 2 : i64,
        destination_group_depth = 2 : i64,
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_dfb {num_tiles = 2 : i64}
          : <[1, 1], !ttcore.tile<32x32, f32>, 4>
          -> tensor<1x2x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x2x!ttcore.tile<32x32, f32>>)
          -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_dfb {num_tiles = 2 : i64}
          : <[1, 1], !ttcore.tile<32x32, f32>, 4>
      %ready = ttl.cb_wait %dst_dfb {num_tiles = 2 : i64}
          : <[1, 1], !ttcore.tile<32x32, f32>, 4>
          -> tensor<1x2x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst_dfb {num_tiles = 2 : i64}
          : <[1, 1], !ttcore.tile<32x32, f32>, 4>
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.pipe_transfer.send %transfer, %src_dfb
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Purpose: a one-shot payload larger than the target's one-packet limit keeps
// completion signaling on the barrier and atomic protocol.
// PLAN: PipeTransport: stream 0 transfer 0 src(0, 0) -> dst(1, 0) to (1, 0) net 0 contract=point_to_point synchronization=receiver_post schedule=grouped credit_completion=immediate group=3
// PLAN-NEXT: PipeTransport:   source blocks=6 block_span=3 stage_depth=2 ownership=dfb scratch_offset=0 scratch_bytes=0 pages=3 page_bytes=4096 loops=0
// PLAN-NEXT: PipeTransport:   endpoint 0 dst(1, 0) DFB 1 block_count=9 slot_span=3 group_depth=3 ownership=dfb scratch_offset=0 scratch_bytes=0 loops=0 address=recurrence(initial=0, stride=3, modulus=9, executions=1)
// PLAN-NEXT: PipeTransport:   completion endpoints=[0] source_reuse=after_completion_group
// LOWERED-LABEL: func.func @oversized_one_shot_transport
// LOWERED-NOT: posted true
// LOWERED: ttkernel.experimental.semaphore_wait(
// LOWERED: ttkernel.noc_async_write %{{.*}}, core
// LOWERED-NEXT: ttkernel.noc_async_write_barrier
// LOWERED: ttkernel.noc_semaphore_inc
// LOWERED-NEXT: ttkernel.noc_async_atomic_barrier
// LOWERED-NOT: posted true
// LOWERED: return
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @oversized_one_shot_transport()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 6} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 6>
    %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 9} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 9>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {
        block_span = 3 : i64,
        destination_group_depth = 3 : i64,
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_dfb {num_tiles = 3 : i64}
          : <[1, 1], !ttcore.tile<32x32, f32>, 9>
          -> tensor<1x3x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x3x!ttcore.tile<32x32, f32>>)
          -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_dfb {num_tiles = 3 : i64}
          : <[1, 1], !ttcore.tile<32x32, f32>, 9>
      %ready = ttl.cb_wait %dst_dfb {num_tiles = 3 : i64}
          : <[1, 1], !ttcore.tile<32x32, f32>, 9>
          -> tensor<1x3x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst_dfb {num_tiles = 3 : i64}
          : <[1, 1], !ttcore.tile<32x32, f32>, 9>
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.pipe_transfer.send %transfer, %src_dfb
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 6>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Purpose: repeated transfers retain accumulated completion signaling even
// when the receiver address recurrence is static and exact.
// PLAN: PipeTransport: stream 0 transfer 0 src(0, 0) -> dst(1, 0) to (1, 0) net 0 contract=point_to_point synchronization=receiver_post schedule=scalar credit_completion=immediate group=1
// PLAN-NEXT: PipeTransport:   source blocks=2 block_span=1 stage_depth=2 ownership=dfb scratch_offset=0 scratch_bytes=0 pages=1 page_bytes=4096 loops=1
// PLAN-NEXT: PipeTransport:   endpoint 0 dst(1, 0) DFB 1 block_count=2 slot_span=1 group_depth=1 ownership=dfb scratch_offset=0 scratch_bytes=0 loops=1 address=recurrence(initial=0, stride=1, modulus=2, executions=2)
// PLAN-NEXT: PipeTransport:   completion endpoints=[0] source_reuse=after_completion_group
// LOWERED-LABEL: func.func @repeated_transport
// LOWERED-NOT: posted true
// LOWERED: ttkernel.noc_async_write_one_packet_set_state
// LOWERED: scf.for
// LOWERED: ttkernel.noc_async_write_one_packet_with_state
// LOWERED-NEXT: ttkernel.noc_async_write_barrier
// LOWERED-NEXT: ttkernel.noc_semaphore_inc
// LOWERED-NEXT: ttkernel.noc_async_atomic_barrier
// LOWERED-NOT: posted true
// LOWERED: return
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @repeated_transport()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.if_dst %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %recv = ttl.cb_reserve %dst_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %token = ttl.pipe_transfer.post %transfer, %recv
            : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.pipe_token<net 0>
        ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
        ttl.cb_push %dst_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        %ready = ttl.cb_wait %dst_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        ttl.cb_pop %dst_dfb
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
      ttl.if_src %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %send = ttl.pipe_transfer.send %transfer, %src_dfb
            : (!ttl.pipe_transfer,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
    }
    func.return
  }
}

// -----

// Purpose: Blackhole's larger one-packet limit admits a three-tile one-shot
// transfer that the target-independent conservative limit rejects above.
// PLAN: PipeTransport: stream 0 transfer 0 src(0, 0) -> dst(1, 0) to (1, 0) net 0 contract=point_to_point synchronization=receiver_post schedule=scalar credit_completion=immediate group=1
// PLAN-NEXT: PipeTransport:   source blocks=2 block_span=1 stage_depth=2 ownership=dfb scratch_offset=0 scratch_bytes=0 pages=3 page_bytes=4096 loops=0
// PLAN-NEXT: PipeTransport:   endpoint 0 dst(1, 0) DFB 1 block_count=2 slot_span=1 group_depth=1 ownership=dfb scratch_offset=0 scratch_bytes=0 loops=0 address=recurrence(initial=0, stride=1, modulus=2, executions=1)
// PLAN-NEXT: PipeTransport:   completion endpoints=[0] source_reuse=after_completion_group
// LOWERED-LABEL: func.func @blackhole_large_one_shot_transport
// LOWERED-NOT: ttkernel.noc_async_write_barrier
// LOWERED: ttkernel.noc_async_write_one_packet_set_state({{.*}}) posted true
// LOWERED: ttkernel.experimental.semaphore_wait(
// LOWERED: ttkernel.noc_async_write_one_packet_with_state({{.*}}) posted true
// LOWERED-NEXT: ttkernel.noc_inline_dw_write({{.*}}) posted true
// LOWERED-NEXT: ttkernel.noc_async_writes_flushed({{.*}}) posted true
// LOWERED-NOT: ttkernel.noc_async_atomic_barrier
// LOWERED: return
module attributes {
  ttl.launch_grid = array<i64: 2, 1>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @blackhole_large_one_shot_transport()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 3], !ttcore.tile<32x32, f32>, 2>
    %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 3], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %reserved = ttl.cb_reserve %dst_dfb
          : <[1, 3], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x3x!ttcore.tile<32x32, f32>>
      %receive = ttl.copy %pipe, %reserved
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x3x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
      ttl.cb_push %dst_dfb
          : <[1, 3], !ttcore.tile<32x32, f32>, 2>
      %ready = ttl.cb_wait %dst_dfb
          : <[1, 3], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x3x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst_dfb
          : <[1, 3], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.copy %src_dfb, %pipe
          : (!ttl.cb<[1, 3], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}
