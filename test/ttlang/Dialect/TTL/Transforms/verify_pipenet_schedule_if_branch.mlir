// RUN: ttlang-opt %s --split-input-file -ttl-verify-pipenet-schedule | FileCheck %s

// Verify valid loopback schedules in mutually exclusive scf.if branches.

// -----

// Each branch posts the receive, sends, then waits for completion. A later
// transfer must follow either branch.
// CHECK-LABEL: func.func @if_branch_no_false_cycle
// CHECK: scf.if
// CHECK: } else {

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @if_branch_no_false_cycle(%runtime_flag: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        {pipeNetName = "net"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    scf.if %runtime_flag {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv_view = ttl.attach_cb %recv_reserve, %recv_cb
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_view
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.wait %recv : !ttl.receive_request
    } else {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv_view = ttl.attach_cb %recv_reserve, %recv_cb
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_view
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.wait %recv : !ttl.receive_request
    }
    %recv_reserve_after = ttl.cb_reserve %recv_cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %recv_view_after = ttl.attach_cb %recv_reserve_after, %recv_cb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %recv_after = ttl.copy %pipe, %recv_view_after
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.receive_request
    %send_after = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send_after : !ttl.transfer_handle<write>
    ttl.wait %recv_after : !ttl.receive_request
    func.return
  }
}

// -----

// Nested branches retain a valid schedule at each leaf.
// CHECK-LABEL: func.func @nested_if_no_false_cycle
// CHECK: scf.if
// CHECK: scf.if
// CHECK: } else {

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @nested_if_no_false_cycle(%flag_a: i1, %flag_b: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        {pipeNetName = "net"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    scf.if %flag_a {
      scf.if %flag_b {
        %recv_reserve = ttl.cb_reserve %recv_cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %recv_view = ttl.attach_cb %recv_reserve, %recv_cb
            : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %recv = ttl.copy %pipe, %recv_view
            : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> !ttl.receive_request
        %send = ttl.copy %send_cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
        ttl.wait %recv : !ttl.receive_request
      } else {
        %recv_reserve = ttl.cb_reserve %recv_cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %recv_view = ttl.attach_cb %recv_reserve, %recv_cb
            : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %recv = ttl.copy %pipe, %recv_view
            : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> !ttl.receive_request
        %send = ttl.copy %send_cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
        ttl.wait %recv : !ttl.receive_request
      }
    }
    func.return
  }
}

// -----

// Branches that call the same helper are separate dynamic occurrences. A
// receiver post in one branch cannot overwrite the post in the other branch.
// CHECK-LABEL: func.func @exclusive_helper_calls
// CHECK: func.call @roundtrip()
// CHECK: } else {
// CHECK: func.call @roundtrip()

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @exclusive_helper_calls(%runtime_flag: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    scf.if %runtime_flag {
      func.call @roundtrip() : () -> ()
    } else {
      func.call @roundtrip() : () -> ()
    }
    func.return
  }

  func.func private @roundtrip() {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        {pipeNetName = "net"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_reserve = ttl.cb_reserve %recv_cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %recv_view = ttl.attach_cb %recv_reserve, %recv_cb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %recv = ttl.copy %pipe, %recv_view
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.receive_request
    %send = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    ttl.wait %recv : !ttl.receive_request
    func.return
  }
}
