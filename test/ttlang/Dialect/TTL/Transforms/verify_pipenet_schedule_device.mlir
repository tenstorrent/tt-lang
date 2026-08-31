// RUN: ttlang-opt %s --split-input-file -ttl-verify-pipenet-schedule | FileCheck %s --enable-var-scope

// Summary: Verify PipeNet occurrence counts at the logical devices where
// source and destination events execute.

// Exact-device predicates select the send at the transfer source and the post
// and wait at its destination.

// CHECK-LABEL: func.func @sender
// CHECK-NEXT: %[[SEND_DFB:.*]] = ttl.bind_cb
// CHECK-NEXT: %[[SEND_PIPE:.*]] = ttl.create_pipe
// CHECK-SAME: source = <coordinates = [0]>
// CHECK-SAME: destination = <coordinates = [3]>
// CHECK-NEXT: %[[IS_SOURCE:.*]] = ttl.is_device <coordinates = [0]>
// CHECK-NEXT: scf.if %[[IS_SOURCE]] {
// CHECK-NEXT: %[[SEND:.*]] = ttl.copy %[[SEND_DFB]], %[[SEND_PIPE]]
// CHECK-NEXT: ttl.wait %[[SEND]]
// CHECK-NEXT: }
// CHECK-NOT: ttl.copy
// CHECK-NEXT: return
// CHECK-LABEL: func.func @receiver
// CHECK-NEXT: %[[RECEIVE_DFB:.*]] = ttl.bind_cb
// CHECK-NEXT: %[[RECEIVE_PIPE:.*]] = ttl.create_pipe
// CHECK-SAME: source = <coordinates = [0]>
// CHECK-SAME: destination = <coordinates = [3]>
// CHECK-NEXT: %[[IS_DESTINATION:.*]] = ttl.is_device <coordinates = [3]>
// CHECK-NEXT: scf.if %[[IS_DESTINATION]] {
// CHECK-NEXT: %[[RESERVED:.*]] = ttl.cb_reserve %[[RECEIVE_DFB]]
// CHECK-NEXT: %[[POST:.*]] = ttl.copy %[[RECEIVE_PIPE]], %[[RESERVED]]
// CHECK-NEXT: ttl.wait %[[POST]]
// CHECK-NEXT: }
// CHECK-NOT: ttl.copy
// CHECK-NEXT: return

#domain = #ttl.device_domain<components = <name = "device", extent = [4]>>
#transfer = #ttl.device_transfer<
    domain = #domain,
    edge = <source = <coordinates = [0]>, destination = <coordinates = [3]>>>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @sender() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #transfer}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %is_source = ttl.is_device <coordinates = [0]> in #domain : i1
    scf.if %is_source {
      %send = ttl.copy %src, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }

  func.func @receiver() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #transfer}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %is_destination = ttl.is_device <coordinates = [3]> in #domain : i1
    scf.if %is_destination {
      %reserved = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post = ttl.copy %pipe, %reserved
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %post : !ttl.receive_request
    }
    func.return
  }
}

// -----

// Selected source and destination properties describe the same transfer row
// from different endpoint callbacks. Record-aware execution counts discard
// the inactive branch before pairing each send with its receiver post.

// CHECK-LABEL: func.func @record_predicate_sender
// CHECK: ttl.pipenet_foreach_src
// CHECK: ttl.selected_pipe_destination_device_index
// CHECK: scf.if
// CHECK: ttl.copy
// CHECK: ttl.copy
// CHECK-LABEL: func.func @record_predicate_receiver
// CHECK: ttl.pipenet_foreach_dst
// CHECK: ttl.selected_pipe_source_device_index
// CHECK: scf.if
// CHECK: ttl.copy
// CHECK: ttl.copy

#record_domain = #ttl.device_domain<
    components = <name = "device", extent = [2]>>
#record_predicate_records = #ttl.pipenet_records<
    net 11 name "record_predicates" pipes [
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <
        domain = #record_domain,
        edge = <source = <coordinates = [0]>,
                destination = <coordinates = [1]>>>>,
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <
        domain = #record_domain,
        edge = <source = <coordinates = [1]>,
                destination = <coordinates = [0]>>>>
]>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @record_predicate_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_zero = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %src_one = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %zero = arith.constant 0 : index
    ttl.pipenet_foreach_src attributes {
        records = #record_predicate_records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %destination = ttl.selected_pipe_destination_device_index %pipe
          : !ttl.selected_pipe_src
      %select_zero = arith.cmpi eq, %destination, %zero : index
      scf.if %select_zero {
        %send = ttl.copy %src_zero, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
               !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      } else {
        %send = ttl.copy %src_one, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
               !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
      ttl.yield
    }
    func.return
  }

  func.func @record_predicate_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dst_zero = ttl.bind_cb {cb_index = 2, block_count = 1}
        {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst_one = ttl.bind_cb {cb_index = 3, block_count = 1}
        {dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %zero = arith.constant 0 : index
    ttl.pipenet_foreach_dst attributes {
        records = #record_predicate_records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %source = ttl.selected_pipe_source_device_index %pipe
          : !ttl.selected_pipe_dst
      %select_zero = arith.cmpi eq, %source, %zero : index
      scf.if %select_zero {
        %reserved = ttl.cb_reserve %dst_zero
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %post = ttl.copy %pipe, %reserved
            : (!ttl.selected_pipe_dst,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.receive_request
        ttl.wait %post : !ttl.receive_request
        ttl.cb_push %dst_zero : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      } else {
        %reserved = ttl.cb_reserve %dst_one
            : <[1, 1], !ttcore.tile<32x32, f32>, 1>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %post = ttl.copy %pipe, %reserved
            : (!ttl.selected_pipe_dst,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.receive_request
        ttl.wait %post : !ttl.receive_request
        ttl.cb_push %dst_one : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      }
      ttl.yield
    }
    func.return
  }
}

// -----

// Nested graph callbacks enumerate only combinations whose selected records
// execute on the same logical device. The sender therefore has one occurrence
// per edge, matching the receiver that has no outer callback.

// CHECK-LABEL: func.func @nested_graph_sender
// CHECK: ttl.pipenet_foreach_src
// CHECK: ttl.pipenet_foreach_src
// CHECK: ttl.copy
// CHECK-LABEL: func.func @nested_graph_receiver
// CHECK: ttl.pipenet_foreach_dst
// CHECK: ttl.copy

#nested_domain = #ttl.device_domain<
    components = <name = "device", extent = [2]>>
#nested_records = #ttl.pipenet_records<net 7 name "nested_graph" pipes [
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <
        domain = #nested_domain,
        edge = <source = <coordinates = [0]>,
                destination = <coordinates = [1]>>>>,
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <
        domain = #nested_domain,
        edge = <source = <coordinates = [1]>,
                destination = <coordinates = [0]>>>>
]>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @nested_graph_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.pipenet_foreach_src attributes {records = #nested_records} {
    ^bb0(%outer_pipe: !ttl.selected_pipe_src):
      ttl.pipenet_foreach_src attributes {records = #nested_records} {
      ^bb0(%inner_pipe: !ttl.selected_pipe_src):
        %send = ttl.copy %src, %inner_pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
               !ttl.selected_pipe_src)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
        ttl.yield
      }
      ttl.yield
    }
    func.return
  }

  func.func @nested_graph_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.pipenet_foreach_dst attributes {records = #nested_records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %reserved = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post = ttl.copy %pipe, %reserved
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %post : !ttl.receive_request
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      ttl.yield
    }
    func.return
  }
}

// -----

// A helper's pipe argument resolves through the active call site before the
// send, post, and wait are specialized to their logical devices.

// CHECK-LABEL: func.func private @device_pipe_helper
// CHECK: %[[HELPER_SOURCE:.*]] = ttl.is_device <coordinates = [0]>
// CHECK-NEXT: scf.if %[[HELPER_SOURCE]] {
// CHECK-NEXT: %[[HELPER_SEND:.*]] = ttl.copy %[[HELPER_SEND_DFB:arg[0-9]+]], %[[HELPER_PIPE:arg[0-9]+]]
// CHECK-NEXT: ttl.wait %[[HELPER_SEND]]
// CHECK-NEXT: }
// CHECK: %[[HELPER_DESTINATION:.*]] = ttl.is_device <coordinates = [3]>
// CHECK-NEXT: scf.if %[[HELPER_DESTINATION]] {
// CHECK-NEXT: %[[HELPER_RESERVED:.*]] = ttl.cb_reserve %[[HELPER_RECEIVE_DFB:arg[0-9]+]]
// CHECK-NEXT: %[[HELPER_POST:.*]] = ttl.copy %[[HELPER_PIPE]], %[[HELPER_RESERVED]]
// CHECK-NEXT: ttl.wait %[[HELPER_POST]]
// CHECK-NEXT: }
// CHECK-NOT: ttl.copy
// CHECK-NEXT: return
// CHECK-LABEL: func.func @helper_callsite
// CHECK: %[[CALL_SEND_DFB:.*]] = ttl.bind_cb
// CHECK-NEXT: %[[CALL_RECEIVE_DFB:.*]] = ttl.bind_cb
// CHECK-NEXT: %[[CALL_PIPE:.*]] = ttl.create_pipe
// CHECK-NEXT: call @device_pipe_helper(%[[CALL_SEND_DFB]], %[[CALL_RECEIVE_DFB]], %[[CALL_PIPE]])
// CHECK-NOT: ttl.copy
// CHECK-NEXT: return

#domain = #ttl.device_domain<components = <name = "device", extent = [4]>>
#transfer = #ttl.device_transfer<
    domain = #domain,
    edge = <source = <coordinates = [0]>, destination = <coordinates = [3]>>>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func private @device_pipe_helper(
      %src: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
      %dst: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
      %pipe: !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>) {
    %is_source = ttl.is_device <coordinates = [0]> in #domain : i1
    scf.if %is_source {
      %send = ttl.copy %src, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    %is_destination = ttl.is_device <coordinates = [3]> in #domain : i1
    scf.if %is_destination {
      %reserved = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post = ttl.copy %pipe, %reserved
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %post : !ttl.receive_request
    }
    func.return
  }

  func.func @helper_callsite()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #transfer}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    func.call @device_pipe_helper(%src, %dst, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>) -> ()
    func.return
  }
}

// -----

// Half-open device ranges select the source and destination without making the
// occurrence counts unknown.

// CHECK-LABEL: func.func @range_sender
// CHECK-NEXT: %[[RANGE_SEND_DFB:.*]] = ttl.bind_cb
// CHECK-NEXT: %[[RANGE_SEND_PIPE:.*]] = ttl.create_pipe
// CHECK-NEXT: %[[SOURCE_RANGE:.*]] = ttl.is_device_in_range
// CHECK-SAME: lo = <coordinates = [0]>
// CHECK-SAME: hi = <coordinates = [1]>
// CHECK-NEXT: scf.if %[[SOURCE_RANGE]] {
// CHECK-NEXT: %[[RANGE_SEND:.*]] = ttl.copy %[[RANGE_SEND_DFB]], %[[RANGE_SEND_PIPE]]
// CHECK-NEXT: ttl.wait %[[RANGE_SEND]]
// CHECK-NEXT: }
// CHECK-NOT: ttl.copy
// CHECK-NEXT: return
// CHECK-LABEL: func.func @range_receiver
// CHECK-NEXT: %[[RANGE_RECEIVE_DFB:.*]] = ttl.bind_cb
// CHECK-NEXT: %[[RANGE_RECEIVE_PIPE:.*]] = ttl.create_pipe
// CHECK-NEXT: %[[DESTINATION_RANGE:.*]] = ttl.is_device_in_range
// CHECK-SAME: lo = <coordinates = [1]>
// CHECK-SAME: hi = <coordinates = [3]>
// CHECK-NEXT: scf.if %[[DESTINATION_RANGE]] {
// CHECK-NEXT: %[[RANGE_RESERVED:.*]] = ttl.cb_reserve %[[RANGE_RECEIVE_DFB]]
// CHECK-NEXT: %[[RANGE_POST:.*]] = ttl.copy %[[RANGE_RECEIVE_PIPE]], %[[RANGE_RESERVED]]
// CHECK-NEXT: ttl.wait %[[RANGE_POST]]
// CHECK-NEXT: }
// CHECK-NOT: ttl.copy
// CHECK-NEXT: return

#domain = #ttl.device_domain<components = <name = "device", extent = [4]>>
#transfer = #ttl.device_transfer<
    domain = #domain,
    edge = <source = <coordinates = [0]>, destination = <coordinates = [2]>>>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @range_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #transfer}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %is_source = ttl.is_device_in_range
        <lo = <coordinates = [0]>, hi = <coordinates = [1]>> in #domain : i1
    scf.if %is_source {
      %send = ttl.copy %src, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }

  func.func @range_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #transfer}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %is_destination = ttl.is_device_in_range
        <lo = <coordinates = [1]>, hi = <coordinates = [3]>> in #domain : i1
    scf.if %is_destination {
      %reserved = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post = ttl.copy %pipe, %reserved
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %post : !ttl.receive_request
    }
    func.return
  }
}
