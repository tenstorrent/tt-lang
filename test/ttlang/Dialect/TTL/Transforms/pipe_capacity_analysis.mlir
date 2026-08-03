// Summary: Pipe capacity analysis debug output reports proven receiver
// endpoints and rejected receiver endpoints before pipe lowering consumes the
// facts.
// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel -debug-only=ttl-pipe-capacity-analysis 2>&1 >/dev/null | FileCheck %s --check-prefix=DEBUG

// Purpose: a point-to-point computed-address stream with a receiver pop is
// proven and records one acquire/release pair.
// DEBUG: PipeCapacity: 1 receiver DFB node(s), 1 receiver endpoint(s)
// DEBUG: PipeCapacity: candidate src(0, 0) -> receiver(1, 0) DFB 1 capacity 2
// DEBUG: PipeCapacity: accept src(0, 0) -> receiver(1, 0) DFB 1 capacity 2: sends=1 pops=1
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @accepted_capacity_edge()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      %ready = ttl.cb_wait %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.pipe_transfer.send %transfer, %src_cb
          : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Purpose: a pop reachable from two wait intervals does not identify one
// completed consumer acquisition and therefore cannot return sender capacity.
// DEBUG: PipeCapacity: 1 receiver DFB node(s), 1 receiver endpoint(s)
// DEBUG: PipeCapacity: candidate src(0, 0) -> receiver(1, 0) DFB 6 capacity 2
// DEBUG-NOT: PipeCapacity: accept src(0, 0) -> receiver(1, 0) DFB 6 capacity 2
// DEBUG: PipeCapacity: reject src(0, 0) -> receiver(1, 0) DFB 6 capacity 2: pop is not owned by a matching receiver wait
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @ambiguous_wait_owner_retains_receiver_post_protocol()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 6, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe
        {expectedReceivers = 1 : i64,
         kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      %ready0 = ttl.cb_wait %dst_cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %ready1 = ttl.cb_wait %dst_cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %alias = tensor.extract_slice %ready0[0, 0] [1, 1] [1, 1]
          : tensor<1x1x!ttcore.tile<32x32, f32>>
          to tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.pipe_transfer.send %transfer, %src_cb
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// A compute thread cannot return capacity with a NoC semaphore increment.
// The transfer therefore retains receiver-post synchronization.
// DEBUG: PipeCapacity: 1 receiver DFB node(s), 1 receiver endpoint(s)
// DEBUG: PipeCapacity: candidate src(0, 0) -> receiver(1, 0) DFB 1 capacity 2
// DEBUG: PipeCapacity: reject src(0, 0) -> receiver(1, 0) DFB 1 capacity 2: pop is not in the receiver NOC domain
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @post_and_send()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe
        {expectedReceivers = 1 : i64,
         kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.pipe_transfer.send %transfer, %src_cb
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }

  func.func @consume()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %ready = ttl.cb_wait %dst_cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    func.return
  }
}

// -----

// Purpose: collective receivers release one logical slot independently, so
// single-counter capacity accounting cannot safely reuse the slot until
// per-receiver releases are tracked.
// DEBUG: PipeCapacity: 2 receiver DFB node(s), 2 receiver endpoint(s)
// DEBUG: PipeCapacity: candidate src(0, 0) -> receiver(1, 0) DFB 5 capacity 2
// DEBUG: PipeCapacity: reject src(0, 0) -> receiver(1, 0) DFB 5 capacity 2: collective capacity-counter synchronization requires per-receiver release accounting
// DEBUG: PipeCapacity: candidate src(0, 0) -> receiver(2, 0) DFB 5 capacity 2
// DEBUG: PipeCapacity: reject src(0, 0) -> receiver(2, 0) DFB 5 capacity 2: collective capacity-counter synchronization requires per-receiver release accounting
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @collective_capacity_uses_receiver_post_synchronization()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 5, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0 {isCollective = true}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {expectedReceivers = 2 : i64, kind = #ttl.pipe_transfer_kind<collective>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> -> !ttl.pipe_transfer
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> {
      %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      %ready = ttl.cb_wait %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> {
      %send = ttl.pipe_transfer.send %transfer, %src_cb
          : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Purpose: without a concrete receiver pop, the release side is not proven.
// This models running the analysis before automatic DFB sync insertion.
// DEBUG: PipeCapacity: 1 receiver DFB node(s), 1 receiver endpoint(s)
// DEBUG: PipeCapacity: candidate src(0, 0) -> receiver(1, 0) DFB 3 capacity 2
// DEBUG-NOT: PipeCapacity: accept src(0, 0) -> receiver(1, 0) DFB 3 capacity 2
// DEBUG: PipeCapacity: reject src(0, 0) -> receiver(1, 0) DFB 3 capacity 2: no matching receiver pops
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @missing_receiver_pop()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      %ready = ttl.cb_wait %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.pipe_transfer.send %transfer, %src_cb
          : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Purpose: when one receiver DFB is written by two transfer nodes, a receiver pop
// names only the DFB, so the analysis cannot attribute the released capacity to
// one sender. Both endpoints are rejected for having more than one writer
// endpoint, the central safety predicate for the capacity protocol.
// DEBUG: PipeCapacity: 1 receiver DFB node(s), 2 receiver endpoint(s)
// DEBUG: PipeCapacity: candidate src(0, 0) -> receiver(2, 0) DFB 2 capacity 2
// DEBUG-NOT: PipeCapacity: accept src(0, 0) -> receiver(2, 0) DFB 2 capacity 2
// DEBUG: PipeCapacity: reject src(0, 0) -> receiver(2, 0) DFB 2 capacity 2: receiver DFB has 2 writer endpoint(s)
// DEBUG: PipeCapacity: candidate src(1, 0) -> receiver(2, 0) DFB 2 capacity 2
// DEBUG-NOT: PipeCapacity: accept src(1, 0) -> receiver(2, 0) DFB 2 capacity 2
// DEBUG: PipeCapacity: reject src(1, 0) -> receiver(2, 0) DFB 2 capacity 2: receiver DFB has 2 writer endpoint(s)
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @two_writers_one_receiver_dfb()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %src_cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pA = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
    %pB = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 1 : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>
    %tA = ttl.pipe_transfer.create %pA {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> -> !ttl.pipe_transfer
    %tB = ttl.pipe_transfer.create %pB {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1> -> !ttl.pipe_transfer
    %recvA = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %tokA = ttl.pipe_transfer.post %tA, %recvA
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
    ttl.pipe_transfer.wait %tokA : !ttl.pipe_token<net 0>
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    %readyA = ttl.cb_wait %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_pop %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    %recvB = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %tokB = ttl.pipe_transfer.post %tB, %recvB
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 1>
    ttl.pipe_transfer.wait %tokB : !ttl.pipe_token<net 1>
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    %readyB = ttl.cb_wait %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_pop %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    ttl.if_src %pA : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
      %sendA = ttl.pipe_transfer.send %tA, %src_cb0
          : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
      ttl.wait %sendA : !ttl.transfer_handle<write>
    }
    ttl.if_src %pB : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1> {
      %sendB = ttl.pipe_transfer.send %tB, %src_cb1
          : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
      ttl.wait %sendB : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Purpose: capacity accounting is one unit per send and one unit per receiver
// pop. A receiver reserve spanning more than one DFB block is therefore
// rejected locally even when the wait/pop pair releases the same multi-block
// span.
// DEBUG: PipeCapacity: 1 receiver DFB node(s), 1 receiver endpoint(s)
// DEBUG: PipeCapacity: candidate src(0, 0) -> receiver(1, 0) DFB 4 capacity 2
// DEBUG-NOT: PipeCapacity: accept src(0, 0) -> receiver(1, 0) DFB 4 capacity 2
// DEBUG: PipeCapacity: reject src(0, 0) -> receiver(1, 0) DFB 4 capacity 2: receiver reserve spans 2 DFB blocks; capacity accounting assumes one
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @multi_block_receive_rejected_for_capacity()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_cb {num_tiles = 2 : i64}
          : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x2x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x2x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_cb {num_tiles = 2 : i64} : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      %ready = ttl.cb_wait %dst_cb {num_tiles = 2 : i64}
          : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x2x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst_cb {num_tiles = 2 : i64} : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.pipe_transfer.send %transfer, %src_cb
          : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}
