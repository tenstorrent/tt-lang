// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -convert-ttl-to-ttkernel

// Summary: Negative tests for ttl-convert-ttl-to-ttkernel block_count checks
// performed by PipeGraph::verifyGatherBlockCounts. The verifier emits two
// distinct wordings: "gather" for unicast pipes converging on one receiver,
// "multicast overlap" for two multicast pipes sharing a destination node.

// Two multicast pipes whose destinations overlap at node (1, 0) each need a
// distinct slot in the receiver CB. With block_count=1 the second pipe's
// assigned slot (1) exceeds the CB capacity.

func.func @multicast_overlap_block_count_too_small()
    attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %p2 = ttl.create_pipe src(2, 0) dst(1, 0) to(1, 3) net 0
      : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>
  %xf1 = ttl.copy %p1, %cb
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
      -> !ttl.transfer_handle
  ttl.wait %xf1 : !ttl.transfer_handle
  // expected-error @below {{multicast overlap pipe receiver CB has block_count=1 but slot 1 is assigned to this pipe; block_count must be >= 2}}
  %xf2 = ttl.copy %p2, %cb
      : (!ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
      -> !ttl.transfer_handle
  ttl.wait %xf2 : !ttl.transfer_handle
  func.return
}
