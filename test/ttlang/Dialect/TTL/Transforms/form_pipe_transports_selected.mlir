// RUN: ttlang-opt %s -ttl-form-pipe-transports | FileCheck %s --check-prefix=FORM
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-form-pipe-transports,convert-ttl-to-ttkernel)' | FileCheck %s --check-prefix=LOWERED

// Summary: Verify transport grouping preserves transfers whose corresponding
// endpoints use both static and record-selected pipe representations.

// The selected source record supplies the send corresponding to the static
// loopback receive. Partial expansion must retain both high-level operations so
// the conversion pass can build their complete transfer graph together.
// FORM-LABEL: func.func @mixed_static_and_selected_transfer
// FORM: %[[PIPE:.*]] = ttl.create_pipe
// FORM: %[[RECEIVE:.*]] = ttl.copy %[[PIPE]], %{{.*}}
// FORM: ttl.pipenet_foreach_src
// FORM: ^bb0(%[[SELECTED:.*]]: !ttl.selected_pipe_src):
// FORM: %[[SEND:.*]] = ttl.copy %{{.*}}, %[[SELECTED]]
// FORM-NEXT: ttl.wait %[[SEND]]
// FORM: ttl.wait %[[RECEIVE]]

// Complete lowering associates the two representations and removes all
// high-level pipe operations without introducing a grouped transport.
// LOWERED-LABEL: func.func @mixed_static_and_selected_transfer
// LOWERED-NOT: ttl.
// LOWERED: ttkernel.noc_async_write
// LOWERED: ttkernel.noc_async_write_barrier
// LOWERED-NOT: ttl.
// LOWERED: return

// The same rule applies when the static operation is the send and the selected
// callback supplies the receiver post.
// FORM-LABEL: func.func @mixed_static_send_and_selected_receive
// FORM: %[[PIPE:.*]] = ttl.create_pipe
// FORM: %[[SEND:.*]] = ttl.copy %{{.*}}, %[[PIPE]]
// FORM-NEXT: ttl.wait %[[SEND]]
// FORM: ttl.pipenet_foreach_dst
// FORM: ^bb0(%[[SELECTED:.*]]: !ttl.selected_pipe_dst):
// FORM: %[[RECEIVE:.*]] = ttl.copy %[[SELECTED]], %{{.*}}
// FORM-NEXT: ttl.wait %[[RECEIVE]]

// LOWERED-LABEL: func.func @mixed_static_send_and_selected_receive
// LOWERED-NOT: ttl.
// LOWERED: ttkernel.noc_async_write
// LOWERED: ttkernel.noc_async_write_barrier
// LOWERED-NOT: ttl.
// LOWERED: return

// A wait-any operation keeps all candidate receives together when one uses a
// selected pipe and another uses a static pipe.
// FORM-LABEL: func.func @wait_any_selected_and_static_candidates
// FORM: %[[STATIC_REQUEST:.*]] = ttl.copy %{{.*}}, %{{.*}}
// FORM: %[[SELECTED_REQUEST:.*]] = ttl.copy %{{.*}}, %{{.*}}
// FORM: ttl.wait_any %[[STATIC_REQUEST]], %[[SELECTED_REQUEST]]
// FORM-NOT: ttl.pipe_transfer.post
// LOWERED-LABEL: func.func @wait_any_selected_and_static_candidates
// LOWERED-NOT: ttl.copy
// LOWERED-NOT: ttl.wait_any
// LOWERED-NOT: ttl.pipe_transfer
// LOWERED: scf.while
// LOWERED-NOT: ttl.copy
// LOWERED-NOT: ttl.wait_any
// LOWERED-NOT: ttl.pipe_transfer
// LOWERED: return
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @mixed_static_and_selected_transfer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %receive_dfb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %reserved = ttl.cb_reserve %receive_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %receive = ttl.copy %pipe, %reserved
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.receive_request
    ttl.pipenet_foreach_src attributes {
        records = #ttl.pipenet_records<net 0 name "loopback" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
                           dstEndX = 0, dstEndY = 0>
        ]>} {
    ^bb0(%selected: !ttl.selected_pipe_src):
      %send = ttl.copy %send_dfb, %selected
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.yield
    }
    ttl.wait %receive : !ttl.receive_request
    func.return
  }

  func.func @mixed_static_send_and_selected_receive()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %send_dfb = ttl.bind_cb {cb_index = 2, block_count = 1} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %receive_dfb = ttl.bind_cb {cb_index = 3, block_count = 1} {dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
    %send = ttl.copy %send_dfb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    ttl.pipenet_foreach_dst attributes {
        records = #ttl.pipenet_records<net 1 name "loopback-receiver" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
                           dstEndX = 0, dstEndY = 0>
        ]>} {
    ^bb0(%selected: !ttl.selected_pipe_dst):
      %reserved = ttl.cb_reserve %receive_dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %receive = ttl.copy %selected, %reserved
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
      ttl.yield
    }
    func.return
  }

  func.func @wait_any_selected_and_static_candidates()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %send_dfb = ttl.bind_cb {cb_index = 4, block_count = 1} {dfb_id = 4 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %static_receive_dfb = ttl.bind_cb {cb_index = 5, block_count = 1} {dfb_id = 5 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %selected_receive_dfb = ttl.bind_cb {cb_index = 6, block_count = 1} {dfb_id = 6 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %static_pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 3
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 3>
    %zero = arith.constant 0 : index
    ttl.pipenet_foreach_dst attributes {
        records = #ttl.pipenet_records<net 2 pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
                           dstEndX = 0, dstEndY = 0>
        ]>} {
    ^bb0(%selected_pipe: !ttl.selected_pipe_dst):
      %static_block = ttl.cb_reserve %static_receive_dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %selected_block = ttl.cb_reserve %selected_receive_dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %static_request = ttl.copy %static_pipe, %static_block
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 3>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      %selected_request = ttl.copy %selected_pipe, %selected_block
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      %static_send = ttl.copy %send_dfb, %static_pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 3>)
          -> !ttl.transfer_handle<write>
      %selected_send = ttl.copy %send_dfb, %selected_pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
             !ttl.selected_pipe_dst)
          -> !ttl.transfer_handle<write>
      ttl.wait %static_send : !ttl.transfer_handle<write>
      ttl.wait %selected_send : !ttl.transfer_handle<write>
      %ready = ttl.wait_any %static_request, %selected_request start %zero
          : (!ttl.receive_request, !ttl.receive_request, index)
          -> !ttl.ready_receive
      %selected = ttl.ready_receive_index %ready : !ttl.ready_receive
      %selected_static = arith.cmpi eq, %selected, %zero : index
      scf.if %selected_static {
        ttl.wait %static_request : !ttl.receive_request
        ttl.cb_push %static_receive_dfb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      } else {
        ttl.wait %selected_request : !ttl.receive_request
        ttl.cb_push %selected_receive_dfb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      }
      ttl.yield
    }
    func.return
  }
}
