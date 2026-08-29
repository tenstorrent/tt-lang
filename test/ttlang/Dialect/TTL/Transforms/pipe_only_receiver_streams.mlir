// Summary: PipeGraph debug output reports why a receiver DFB producer stream
// is or is not proven pipe-only before pipe capacity analysis consumes the
// graph fact.
// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel -debug-only=ttl-pipe-graph 2>&1 >/dev/null | FileCheck %s --check-prefix=GRAPH

// GRAPH: PipeGraph: accept pipe-only producer stream for receiver(1, 0) DFB 1
// GRAPH: PipeGraph: reject pipe-only producer stream for receiver(1, 0) DFB 1: push block count does not match posted receiver slot span
// GRAPH: PipeGraph: reject pipe-only producer stream for receiver(1, 0) DFB 1: push is not in a receiver NOC thread
// GRAPH: PipeGraph: accept pipe-only producer stream for receiver(1, 0) DFB 1
// GRAPH: PipeGraph: reject pipe-only producer stream for receiver(1, 0) DFB 1: push reserve owns no matching receiver post
// GRAPH: PipeGraph: accept pipe-only producer stream for receiver(1, 0) DFB 1
// GRAPH: PipeGraph: reject pipe-only producer stream for receiver(1, 0) DFB 1: post is not consumed by a receiver push
// GRAPH: PipeGraph: accept pipe-only producer stream for receiver(1, 0) DFB 1
// GRAPH: PipeGraph: reject pipe-only producer stream for receiver(1, 0) DFB 1: push has no unique receiver reserve owner

// Purpose: the canonical one-post, one-push producer stream is proven
// pipe-only.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @accepted_pipe_only_stream()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {kind = #ttl.pipe_transfer_kind<point_to_point>}
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

// Purpose: receiver posts from one DFB reservation cannot publish more blocks
// than the subsequent push advances.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @one_reserve_with_multiple_posts()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %pipe1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 1 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1>
    %transfer0 = ttl.pipe_transfer.create %pipe0 {kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
    %transfer1 = ttl.pipe_transfer.create %pipe1 {kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1> -> !ttl.pipe_transfer
    ttl.if_dst %pipe0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token0 = ttl.pipe_transfer.post %transfer0, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
      %token1 = ttl.pipe_transfer.post %transfer1, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 1>
      ttl.pipe_transfer.wait %token0 : !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token1 : !ttl.pipe_token<net 1>
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_src %pipe0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send0 = ttl.pipe_transfer.send %transfer0, %src_cb
          : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
      ttl.wait %send0 : !ttl.transfer_handle<write>
    }
    ttl.if_src %pipe1 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1> {
      %send1 = ttl.pipe_transfer.send %transfer1, %src_cb
          : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
      ttl.wait %send1 : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Purpose: only a receiver NOC thread can publish a PipeNet destination DFB.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @receiver_push_in_compute_thread()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
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

// Purpose: a node-selected receiver wrapper does not interrupt the proven
// post, completion wait, and push order on that receiver.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @receiver_wait_inside_role_wrapper()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
    %is_dst = ttl.is_dst {pipe_net_id = 0 : i64}
    scf.if %is_dst {
      %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %token = ttl.pipe_transfer.post %transfer, %recv
            : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
        ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      }
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
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

// Purpose: an unrelated receiver-domain push on the receiver DFB prevents the
// stream from being pipe-only.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @stray_receiver_push()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      %ready = ttl.cb_wait %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      %stray = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
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

// Purpose: consumer wait and pop counts do not affect the producer write
// pointer proof. Capacity analysis validates their release accounting.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @wait_pop_count_mismatch()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      %ready = ttl.cb_wait %dst_cb {num_tiles = 2 : i64} : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x2x!ttcore.tile<32x32, f32>>
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

// Purpose: a receiver post that never reaches a receiver DFB push leaves the
// stream unproven.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @post_never_pushed()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
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

// Purpose: consumer wait/pop operations do not affect the producer write
// pointer proof.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @consumer_wait_pop_does_not_change_producer_proof()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      %ready = ttl.cb_wait %dst_cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
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

// Purpose: a receiver push reachable from two receiver reserves has ambiguous
// ownership, so the stream is not proven pipe-only.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @push_with_ambiguous_reserve_owner()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %unused = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %alias = tensor.extract_slice %unused[0, 0] [1, 1] [1, 1]
          : tensor<1x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
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
