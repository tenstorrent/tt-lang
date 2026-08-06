// Summary: Pipe transport planning records backend-independent scalar
// schedules, endpoint storage, receiver address recurrences, and completion
// groups.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=false})' -debug-only=ttl-pipe-transport-plan 2>&1 >/dev/null | FileCheck %s --check-prefix=PLAN

// Purpose: a point-to-point stream records one logical transfer per group and
// releases source storage after its only endpoint completes.
// PLAN: PipeTransport: stream 0 transfer 0 src(0, 0) -> dst(1, 0) to (1, 0) net 0 contract=point_to_point synchronization=receiver_post schedule=scalar credit_completion=immediate group=1
// PLAN-NEXT: PipeTransport:   source blocks=2 block_span=1 stage_depth=2 ownership=dfb scratch_offset=0 scratch_bytes=0 pages=1 page_bytes=4096 loops=0
// PLAN-NEXT: PipeTransport:   endpoint 0 dst(1, 0) DFB 1 block_count=2 slot_span=1 group_depth=1 ownership=dfb scratch_offset=0 scratch_bytes=0 loops=0 address=recurrence(initial=0, stride=1, modulus=2, executions=1)
// PLAN-NEXT: PipeTransport:   completion endpoints=[0] source_reuse=after_completion_group
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @point_to_point_transport()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
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
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @collective_transport()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_dfb = ttl.bind_cb {cb_index = 5, block_count = 2}
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
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @grouped_transport()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 4}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>
    %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 4}
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
