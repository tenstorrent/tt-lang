// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -convert-ttl-to-ttkernel

// Summary: Negative tests for pipe receiver DFB validation and pipe synchronization
// resource diagnostics in ttl-convert-ttl-to-ttkernel.

// Two unicast pipes converging on node (1, 0) need distinct slots in the
// receiver DFB. With block_count=1 the second pipe's assigned slot exceeds the
// DFB capacity.

func.func @gather_block_count_too_small()
    attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p2 = ttl.create_pipe src(2, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 0) net 0>
  %recv1 = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf1 = ttl.copy %p1, %recv1
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %xf1 : !ttl.transfer_handle
  %recv2 = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{gather pipe receiver DFB has block_count=1 but slot 1 is assigned to this pipe; block_count must be >= 2}}
  %xf2 = ttl.copy %p2, %recv2
      : (!ttl.pipe<src(2, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %xf2 : !ttl.transfer_handle
  func.return
}

// -----

// A collective pipe cannot publish different receiver DFB slice offsets until
// per-receiver destination addresses are implemented.

func.func @collective_destination_addresses_differ_by_destination()
    attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
  %recv_group = ttl.cb_reserve %cb
      : <[1, 2], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x2x!ttcore.tile<32x32, f32>>
  %recv0 = tensor.extract_slice %recv_group[0, 0] [1, 1] [1, 1]
      : tensor<1x2x!ttcore.tile<32x32, f32>>
      to tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-note @below {{previous collective receive post for this pipe was here}}
  %xf0 = ttl.copy %p, %recv0
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %xf0 : !ttl.transfer_handle
  %recv1 = tensor.extract_slice %recv_group[0, 1] [1, 1] [1, 1]
      : tensor<1x2x!ttcore.tile<32x32, f32>>
      to tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{collective pipe receive posts publish different destination addresses; per-receiver destination addresses are tracked by issue #617}}
  %xf1 = ttl.copy %p, %recv1
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %xf1 : !ttl.transfer_handle
  func.return
}

// -----

// Collective destination addresses must be statically traceable until
// per-receiver destination addresses are represented explicitly.

func.func @collective_destination_address_dynamic_offset_rejected(%offset: index)
    attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
  %recv_group = ttl.cb_reserve %cb
      : <[1, 2], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x2x!ttcore.tile<32x32, f32>>
  %recv = tensor.extract_slice %recv_group[0, %offset] [1, 1] [1, 1]
      : tensor<1x2x!ttcore.tile<32x32, f32>>
      to tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{collective pipe destination address could not be determined statically; per-receiver destination addresses are tracked by issue #617}}
  %xf = ttl.copy %p, %recv
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %xf : !ttl.transfer_handle
  func.return
}

// -----

// Receiver completion still uses local semaphore ids. A PipeNet id above the
// local limit is rejected even when sender-ready counters use GlobalSemaphore
// allocation.

// expected-error @below {{pipe synchronization requires 17 hardware semaphore ids, exceeding TT hardware limit of 16; issue #619 tracks scalable pipe synchronization allocation}}
// expected-note @below {{highest allocated semaphore id is 16 for receiver-completion counter}}
module {
  func.func @unicast_pipe_sync_exceeds_hardware_semaphore_limit()
      attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %p2 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
    %p3 = ttl.create_pipe src(0, 0) dst(3, 0) to(3, 0) net 0
        : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>
    %p4 = ttl.create_pipe src(0, 0) dst(4, 0) to(4, 0) net 0
        : !ttl.pipe<src(0, 0) dst(4, 0) to(4, 0) net 0>
    %p5 = ttl.create_pipe src(0, 0) dst(5, 0) to(5, 0) net 0
        : !ttl.pipe<src(0, 0) dst(5, 0) to(5, 0) net 0>
    %p6 = ttl.create_pipe src(0, 0) dst(6, 0) to(6, 0) net 0
        : !ttl.pipe<src(0, 0) dst(6, 0) to(6, 0) net 0>
    %p7 = ttl.create_pipe src(0, 0) dst(7, 0) to(7, 0) net 0
        : !ttl.pipe<src(0, 0) dst(7, 0) to(7, 0) net 0>
    %p8 = ttl.create_pipe src(0, 0) dst(8, 0) to(8, 0) net 0
        : !ttl.pipe<src(0, 0) dst(8, 0) to(8, 0) net 0>
    %p9 = ttl.create_pipe src(0, 0) dst(9, 0) to(9, 0) net 0
        : !ttl.pipe<src(0, 0) dst(9, 0) to(9, 0) net 0>
    %p10 = ttl.create_pipe src(0, 0) dst(10, 0) to(10, 0) net 0
        : !ttl.pipe<src(0, 0) dst(10, 0) to(10, 0) net 0>
    %p11 = ttl.create_pipe src(0, 0) dst(11, 0) to(11, 0) net 0
        : !ttl.pipe<src(0, 0) dst(11, 0) to(11, 0) net 0>
    %p12 = ttl.create_pipe src(0, 0) dst(12, 0) to(12, 0) net 0
        : !ttl.pipe<src(0, 0) dst(12, 0) to(12, 0) net 0>
    %p13 = ttl.create_pipe src(0, 0) dst(13, 0) to(13, 0) net 0
        : !ttl.pipe<src(0, 0) dst(13, 0) to(13, 0) net 0>
    %p14 = ttl.create_pipe src(0, 0) dst(14, 0) to(14, 0) net 0
        : !ttl.pipe<src(0, 0) dst(14, 0) to(14, 0) net 0>
    %p15 = ttl.create_pipe src(0, 0) dst(15, 0) to(15, 0) net 0
        : !ttl.pipe<src(0, 0) dst(15, 0) to(15, 0) net 0>
    %p16 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 16
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 16>
    %recv1 = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf1 = ttl.copy %p1, %recv1
        : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf1 : !ttl.transfer_handle
    %recv2 = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf2 = ttl.copy %p2, %recv2
        : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf2 : !ttl.transfer_handle
    %recv3 = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf3 = ttl.copy %p3, %recv3
        : (!ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf3 : !ttl.transfer_handle
    %recv4 = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf4 = ttl.copy %p4, %recv4
        : (!ttl.pipe<src(0, 0) dst(4, 0) to(4, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf4 : !ttl.transfer_handle
    %recv5 = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf5 = ttl.copy %p5, %recv5
        : (!ttl.pipe<src(0, 0) dst(5, 0) to(5, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf5 : !ttl.transfer_handle
    %recv6 = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf6 = ttl.copy %p6, %recv6
        : (!ttl.pipe<src(0, 0) dst(6, 0) to(6, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf6 : !ttl.transfer_handle
    %recv7 = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf7 = ttl.copy %p7, %recv7
        : (!ttl.pipe<src(0, 0) dst(7, 0) to(7, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf7 : !ttl.transfer_handle
    %recv8 = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf8 = ttl.copy %p8, %recv8
        : (!ttl.pipe<src(0, 0) dst(8, 0) to(8, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf8 : !ttl.transfer_handle
    func.return
  }
}

// -----

// Two collective pipes whose destinations overlap at node (1, 0) each need a
// distinct slot in the receiver DFB. With block_count=1 the second pipe's
// assigned slot (1) exceeds the DFB capacity.

func.func @collective_overlap_block_count_too_small()
    attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %p2 = ttl.create_pipe src(2, 0) dst(1, 0) to(1, 3) net 0
      : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>
  %recv1 = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf1 = ttl.copy %p1, %recv1
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %xf1 : !ttl.transfer_handle
  %recv2 = ttl.cb_reserve %cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{collective overlap pipe receiver DFB has block_count=1 but slot 1 is assigned to this pipe; block_count must be >= 2}}
  %xf2 = ttl.copy %p2, %recv2
      : (!ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %xf2 : !ttl.transfer_handle
  func.return
}
