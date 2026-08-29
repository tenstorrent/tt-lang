// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -ttl-verify-pipenet-schedule

// Summary: Negative tests for invalid or unprovable PipeNet synchronization
// schedules.

// Schedule verification requires the launch grid used to specialize events to
// individual kernel instances.

// expected-error @below {{ttl-verify-pipenet-schedule requires a `ttl.launch_grid` module attribute}}
module {
  func.func @missing_launch_grid()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    func.return
  }
}

// -----

// Pipe endpoints must refer to cores instantiated by the module launch grid.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @source_outside_launch_grid()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{declares source core_x=2, core_y=0 outside the module `ttl.launch_grid`}}
    %pipe = ttl.create_pipe src(2, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 0) net 0>
    func.return
  }
}

// -----

// Wait-any blocks when every candidate send follows it in the same thread.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @wait_any_all_sends_after()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %landing0 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %landing1 = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
    %dst0 = ttl.cb_reserve %landing0
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %dst1 = ttl.cb_reserve %landing1
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %request0 = ttl.copy %pipe0, %dst0
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.receive_request
    %request1 = ttl.copy %pipe1, %dst1
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.receive_request
    %start = arith.constant 0 : index
    // expected-error @below {{receive wait-any can block with every candidate send ordered after the selection at core_x=0, core_y=0}}
    %ready = ttl.wait_any %request0, %request1 start %start
        : (!ttl.receive_request, !ttl.receive_request, index)
        -> !ttl.ready_receive
    // expected-note @below {{this candidate send cannot complete before the wait-any}}
    %send0 = ttl.copy %source, %pipe0
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send0 : !ttl.transfer_handle<write>
    // expected-note @below {{this candidate send cannot complete before the wait-any}}
    %send1 = ttl.copy %source, %pipe1
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send1 : !ttl.transfer_handle<write>
    func.return
  }
}

// -----

// Endpoint validation precedes destination-domain enumeration, so a malformed
// range cannot make verification time proportional to that range.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @destination_outside_launch_grid()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{declares destination range core_x=1..100000000, core_y=0..0 outside the module `ttl.launch_grid`}}
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(100000000, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(100000000, 0) net 0>
    func.return
  }
}

// -----

// Record-backed PipeNet sources must refer to cores instantiated by the module
// launch grid.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @record_source_outside_launch_grid()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{declares source core_x=2, core_y=0 outside the module `ttl.launch_grid`}}
    ttl.pipenet_foreach_src attributes {
        records = #ttl.pipenet_records<net 0 name "invalid_source" pipes [
          #ttl.pipe_record<srcX = 2, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      ttl.yield
    }
    func.return
  }
}

// -----

// Record-backed PipeNet destination ranges must remain inside the module
// launch grid.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @record_destination_outside_launch_grid()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{declares destination range core_x=1..2, core_y=0..0 outside the module `ttl.launch_grid`}}
    ttl.pipenet_foreach_dst attributes {
        records = #ttl.pipenet_records<net 0 name "invalid_destination" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 2, dstEndY = 0, isCollective = true>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      ttl.yield
    }
    func.return
  }
}

// -----

// A collective send missing every receiver post produces one primary error.
// Notes identify the additional receiver coordinates with the same mismatch.

module attributes {ttl.launch_grid = [3 : i64, 1 : i64]} {
  func.func @collective_missing_receiver_posts()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0> {
      // expected-error @below {{requires one static receiver post definition for each static send definition at receiver core_x=1, core_y=0}}
      // expected-note @below {{this send has no corresponding receiver post}}
      // expected-note @below {{the same mismatch applies at receiver core_x=2, core_y=0}}
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Schedule correspondence pairs static event definitions. It conservatively
// rejects mutually exclusive posts when neither branch is proven not to run.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @conditional_receiver_posts(%condition: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      scf.if %condition {
        %reserve = ttl.cb_reserve %recv_cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %receive = ttl.copy %pipe, %reserve
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> !ttl.receive_request
      } else {
        %reserve = ttl.cb_reserve %recv_cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        // expected-error @below {{found 2 static receiver post definition(s) and 1 static send definition(s)}}
        // expected-note @below {{this receiver post has no corresponding send}}
        %receive = ttl.copy %pipe, %reserve
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> !ttl.receive_request
      }
    }
    func.return
  }
}

// -----

// Independent kernel threads have no program order. Their send definitions
// cannot share one pipe endpoint's synchronization state.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @first_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-note @below {{the first send definition is in @first_sender}}
    %send = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    func.return
  }

  func.func @second_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{send definitions for the same pipe endpoint occur in multiple kernel-thread functions}}
    %send = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    func.return
  }
}

// -----

// A five-node forward ring has a ten-edge wait-for cycle. The diagnostic
// reports the generic cycle and limits its explanation to eight edge notes.

module attributes {ttl.launch_grid = [5 : i64, 1 : i64]} {
  func.func @forward_ring_cycle()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %pipe1 = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 1
        : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>
    %pipe2 = ttl.create_pipe src(2, 0) dst(3, 0) to(3, 0) net 2
        : !ttl.pipe<src(2, 0) dst(3, 0) to(3, 0) net 2>
    %pipe3 = ttl.create_pipe src(3, 0) dst(4, 0) to(4, 0) net 3
        : !ttl.pipe<src(3, 0) dst(4, 0) to(4, 0) net 3>
    %pipe4 = ttl.create_pipe src(4, 0) dst(0, 0) to(0, 0) net 4
        : !ttl.pipe<src(4, 0) dst(0, 0) to(0, 0) net 4>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

    ttl.if_src %pipe0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // expected-error @below {{pipe schedule contains a wait-for cycle on PipeNet net_0}}
      %send = ttl.copy %send_cb, %pipe0
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    ttl.if_src %pipe1
        : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1> {
      // expected-note @below {{sender waits for receiver post at core_x=2, core_y=0 before send at core_x=1, core_y=0}}
      %send = ttl.copy %send_cb, %pipe1
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>)
          -> !ttl.transfer_handle<write>
    }
    ttl.if_src %pipe2
        : !ttl.pipe<src(2, 0) dst(3, 0) to(3, 0) net 2> {
      // expected-note @below {{sender waits for receiver post at core_x=3, core_y=0 before send at core_x=2, core_y=0}}
      %send = ttl.copy %send_cb, %pipe2
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(2, 0) dst(3, 0) to(3, 0) net 2>)
          -> !ttl.transfer_handle<write>
    }
    ttl.if_src %pipe3
        : !ttl.pipe<src(3, 0) dst(4, 0) to(4, 0) net 3> {
      // expected-note @below {{sender waits for receiver post at core_x=4, core_y=0 before send at core_x=3, core_y=0}}
      %send = ttl.copy %send_cb, %pipe3
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(3, 0) dst(4, 0) to(4, 0) net 3>)
          -> !ttl.transfer_handle<write>
    }
    ttl.if_src %pipe4
        : !ttl.pipe<src(4, 0) dst(0, 0) to(0, 0) net 4> {
      // expected-note @below {{sender waits for receiver post at core_x=0, core_y=0 before send at core_x=4, core_y=0}}
      %send = ttl.copy %send_cb, %pipe4
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(4, 0) dst(0, 0) to(0, 0) net 4>)
          -> !ttl.transfer_handle<write>
    }

    ttl.if_dst %pipe0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %receive = ttl.copy %pipe0, %reserve
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
    }
    ttl.if_dst %pipe1
        : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1> {
      %reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-note @below {{program order requires receiver post at core_x=2, core_y=0 after send at core_x=2, core_y=0}}
      %receive = ttl.copy %pipe1, %reserve
          : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
    }
    ttl.if_dst %pipe2
        : !ttl.pipe<src(2, 0) dst(3, 0) to(3, 0) net 2> {
      %reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-note @below {{program order requires receiver post at core_x=3, core_y=0 after send at core_x=3, core_y=0}}
      %receive = ttl.copy %pipe2, %reserve
          : (!ttl.pipe<src(2, 0) dst(3, 0) to(3, 0) net 2>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
    }
    ttl.if_dst %pipe3
        : !ttl.pipe<src(3, 0) dst(4, 0) to(4, 0) net 3> {
      %reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-note @below {{program order requires receiver post at core_x=4, core_y=0 after send at core_x=4, core_y=0}}
      %receive = ttl.copy %pipe3, %reserve
          : (!ttl.pipe<src(3, 0) dst(4, 0) to(4, 0) net 3>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
    }
    ttl.if_dst %pipe4
        : !ttl.pipe<src(4, 0) dst(0, 0) to(0, 0) net 4> {
      %reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-note @below {{program order requires receiver post at core_x=0, core_y=0 after send at core_x=0, core_y=0}}
      %receive = ttl.copy %pipe4, %reserve
          : (!ttl.pipe<src(4, 0) dst(0, 0) to(0, 0) net 4>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
    }
    func.return
  }
}

// -----

// Block storage order does not define CFG execution order. Reject multi-block
// functions instead of constructing incorrect program-order edges.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  // expected-error @below {{cannot verify PipeNet synchronization in multi-block function @cfg_block_order}}
  func.func @cfg_block_order()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    cf.br ^post
  ^send:
    %send = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    cf.br ^exit
  ^post:
    %reserve = ttl.cb_reserve %recv_cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %receive = ttl.copy %pipe, %reserve
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.receive_request
    cf.br ^send
  ^exit:
    func.return
  }
}

// -----

// Nested regions also require one block when they contain pipe events. Block
// storage order does not define the execution order of the nested CFG.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @nested_cfg_block_order()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{cannot verify PipeNet synchronization in a multi-block region of this operation}}
    scf.execute_region {
      cf.br ^post
    ^send:
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      cf.br ^exit
    ^post:
      %reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %receive = ttl.copy %pipe, %reserve
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      cf.br ^send
    ^exit:
      ttl.wait %receive : !ttl.receive_request
      scf.yield
    }
    func.return
  }
}

// -----

// A call to a helper with pipe events contributes those events to the caller's
// schedule, so its enclosing region must also have one block.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func private @nested_cfg_send(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>) {
    %send = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    func.return
  }

  func.func @nested_cfg_call()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reserve = ttl.cb_reserve %recv_cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %receive = ttl.copy %pipe, %reserve
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.receive_request
    // expected-error @below {{cannot verify PipeNet synchronization in a multi-block region of this operation}}
    scf.execute_region {
      cf.br ^invoke
    ^exit:
      scf.yield
    ^invoke:
      func.call @nested_cfg_send(%send_cb, %pipe)
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>) -> ()
      cf.br ^exit
    }
    func.return
  }
}

// -----

// A coordinate-dependent condition with an unevaluable operand makes the
// receiver-post domain unknown. The schedule pass must not omit that event.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unknown_receiver_domain(%offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %sum = arith.addi %core_x, %offset : index
    %c1 = arith.constant 1 : index
    // expected-note @below {{this coordinate-dependent condition cannot be evaluated statically}}
    %is_receiver = arith.cmpi eq, %sum, %c1 : index
    scf.if %is_receiver {
      %reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-error @below {{cannot verify PipeNet synchronization because this receiver post has an unknown launch-node domain}}
      %receive = ttl.copy %pipe, %reserve
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
    }
    func.return
  }
}

// -----

// A helper argument can preserve a coordinate-dependent loop bound across a
// call. The source executes once, while the destination executes twice.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func private @coordinate_dependent_count_helper(
      %count: index,
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %recv_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %iteration = %c0 to %count step %c1 {
      ttl.if_src %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        // expected-error @below {{cannot prove a one-to-one synchronization schedule}}
        %send = ttl.copy %send_cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
      ttl.if_dst %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %reserve = ttl.cb_reserve %recv_cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        // expected-note @below {{matching receiver post occurrence is here}}
        %receive = ttl.copy %pipe, %reserve
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> !ttl.receive_request
        ttl.wait %receive : !ttl.receive_request
      }
    }
    func.return
  }

  func.func @coordinate_dependent_count_through_call()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %c1 = arith.constant 1 : index
    %count = arith.addi %core_x, %c1 : index
    func.call @coordinate_dependent_count_helper(
        %count, %send_cb, %recv_cb, %pipe)
        : (index, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.return
  }
}

// -----

// Pipe events must be reachable from a kernel-thread function so their launch
// nodes and call sites are defined.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unreachable_pipe_event() {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // expected-error @below {{cannot verify PipeNet synchronization in @unreachable_pipe_event because it is not reachable from a kernel-thread function through direct calls}}
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// PipeNet references are validated even when the module has no declaration.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @undeclared_pipe_predicate()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{references unknown PipeNet net_7}}
    %is_src = ttl.is_src {pipe_net_id = 7 : i64}
    scf.if %is_src {
    }
    func.return
  }
}

// -----

// A same-thread loopback receive wait before the matching send creates a
// cycle: the wait needs the send to complete, but program order places the
// send after the wait.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @loopback_wait_before_send() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        {pipeNetName = "net"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
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
      // expected-error @below {{receive wait occurs before the send that completes it on PipeNet net}}
      // expected-note @below {{this wait blocks until the sender transfers into the posted destination dataflow buffer slot}}
      // expected-note @below {{move the receive wait after the send, or place send and receive in separate data-movement threads}}
      ttl.wait %recv : !ttl.receive_request
      // expected-note @below {{this send is ordered after the wait in the same data-movement thread}}
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// A send cannot rendezvous when no receiver posts a destination DFB slot.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @missing_receiver_post()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // expected-error @below {{PipeNet net_0 requires one static receiver post definition for each static send definition at receiver core_x=1, core_y=0; found 0 static receiver post definition(s) and 1 static send definition(s)}}
      // expected-note @below {{this send has no corresponding receiver post}}
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// A receiver post and wait cannot complete when the corresponding send is
// absent.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @missing_send()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-note @below {{defining receiver post is here}}
      %recv = ttl.copy %pipe, %recv_reserve
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      // expected-error @below {{receive wait has no send corresponding to its defining receiver post on PipeNet net_0 at core_x=1, core_y=0}}
      ttl.wait %recv : !ttl.receive_request
    }
    func.return
  }
}

// -----

// A wait-any candidate cannot complete without a corresponding send.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @wait_any_missing_send()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_reserve
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      %start = arith.constant 0 : index
      // expected-error @below {{receive wait-any has no candidate send corresponding to a defining receiver post at core_x=1, core_y=0}}
      // expected-error @below {{receive wait-any can block with every candidate send ordered after the selection at core_x=1, core_y=0}}
      %ready = ttl.wait_any %recv start %start
          : (!ttl.receive_request, index) -> !ttl.ready_receive
    }
    func.return
  }
}

// -----

// Recursive helper calls containing pipe events have no finite static event
// sequence for correspondence and program-order analysis.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func private @recursive_send(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    %send = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    // expected-error @below {{cannot verify PipeNet synchronization through a recursive call to @recursive_send}}
    func.call @recursive_send(%send_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.return
  }

  func.func @recursive_pipe_events()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      func.call @recursive_send(%send_cb, %pipe)
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    }
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %receive = ttl.copy %pipe, %reserve
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
    }
    func.return
  }
}

// -----

// Two calls to a send helper are two static send occurrences. One receiver
// post cannot satisfy both calls.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func private @send_twice_helper(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    // expected-error @below {{PipeNet net_0 requires one static receiver post definition for each static send definition at receiver core_x=1, core_y=0; found 1 static receiver post definition(s) and 2 static send definition(s)}}
    // expected-note @below {{this send has no corresponding receiver post}}
    %send = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    func.return
  }

  func.func @helper_call_count_mismatch()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      func.call @send_twice_helper(%send_cb, %pipe)
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
      func.call @send_twice_helper(%send_cb, %pipe)
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    }
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_reserve
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      ttl.wait %recv : !ttl.receive_request
    }
    func.return
  }
}

// -----

// Expanding a helper at its call site exposes a loopback send that executes
// before the caller posts the receiver reservation.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func private @loopback_send_helper(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>) {
    // expected-error @below {{pipe send occurs before the receiver posts a dataflow buffer reservation on PipeNet loopback}}
    // expected-note @below {{this send waits for each destination to post `ttl.copy(pipe, dst)`}}
    // expected-note @below {{move `ttl.copy(pipe, dst)` before the dependent send, or place send and receive in separate data-movement threads}}
    %send = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    func.return
  }

  func.func @helper_send_before_caller_post()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        {pipeNetName = "loopback"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      func.call @loopback_send_helper(%send_cb, %pipe)
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>) -> ()
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-note @below {{this receiver post is ordered after the send in the same data-movement thread}}
      %recv = ttl.copy %pipe, %recv_reserve
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      ttl.wait %recv : !ttl.receive_request
    }
    func.return
  }
}

// -----

// Each foreach callback completes before the next matching record begins.
// Reversing the sender records creates a cycle between the first receiver wait
// and the sender's first selected record. The sender callback is in a helper to
// verify that call expansion preserves the enclosing record order.

module attributes {ttl.launch_grid = [2 : i64, 2 : i64]} {
  func.func @receiver_record_order()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.pipenet_foreach_dst attributes {
        records = #ttl.pipenet_records<net 0 name "ordered" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0, isCollective = true>,
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 1, isCollective = true>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-note @below {{program order requires receiver post at core_x=1, core_y=0 after receive completion at core_x=1, core_y=0}}
      %recv = ttl.copy %pipe, %recv_reserve
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      // expected-error @below {{pipe schedule contains a wait-for cycle on PipeNet ordered}}
      // expected-note @below {{receive completion at core_x=1, core_y=0 waits for send at core_x=0, core_y=0 to transfer data}}
      ttl.wait %recv : !ttl.receive_request
      ttl.cb_push %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      ttl.yield
    }
    func.return
  }

  func.func private @send_records_reversed(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) {
    ttl.pipenet_foreach_src attributes {
        records = #ttl.pipenet_records<net 0 name "ordered" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 1, isCollective = true>,
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0, isCollective = true>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      // expected-note @below {{sender waits for receiver post at core_x=1, core_y=0 before send at core_x=0, core_y=0}}
      // expected-note @below {{program order requires send at core_x=0, core_y=0 after send at core_x=0, core_y=0}}
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.yield
    }
    func.return
  }

  func.func @sender_reversed_record_order()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.call @send_records_reversed(%send_cb)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
    func.return
  }
}

// -----

// Straight-line post and send counts must agree for one logical pipe.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @different_static_post_send_counts()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_reserve
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      ttl.wait %recv : !ttl.receive_request
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send0 = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send0 : !ttl.transfer_handle<write>
      // expected-error @below {{PipeNet net_0 requires one static receiver post definition for each static send definition at receiver core_x=1, core_y=0; found 1 static receiver post definition(s) and 2 static send definition(s)}}
      // expected-note @below {{this send has no corresponding receiver post}}
      %send1 = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send1 : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// A send operation inside a loop does not correspond to a receiver-post
// operation outside the loop because their dynamic execution counts differ.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @different_rendezvous_loop_contexts()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-note @below {{matching receiver post occurrence is here}}
      %recv = ttl.copy %pipe, %recv_reserve
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      ttl.wait %recv : !ttl.receive_request
    }
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        // expected-error @below {{cannot prove a one-to-one synchronization schedule on PipeNet net_0 for receiver core_x=1, core_y=0; receiver post and send occurrences do not have matching proven execution counts and conditions}}
        %send = ttl.copy %send_cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
    }
    func.return
  }
}

// -----

// The source and destination evaluate the shared scf.for upper bound at their
// own coordinates. One send cannot match two receiver posts.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @node_dependent_rendezvous_count()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %upper = arith.addi %core_x, %c1 : index
    scf.for %iteration = %c0 to %upper step %c1 {
      ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        // expected-error @below {{cannot prove a one-to-one synchronization schedule on PipeNet net_0 for receiver core_x=1, core_y=0; receiver post and send occurrences do not have matching proven execution counts and conditions}}
        %send = ttl.copy %send_cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
      ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %recv_reserve = ttl.cb_reserve %recv_cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        // expected-note @below {{matching receiver post occurrence is here}}
        %recv = ttl.copy %pipe, %recv_reserve
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> !ttl.receive_request
        ttl.wait %recv : !ttl.receive_request
      }
    }
    func.return
  }
}

// -----

// Equal unresolved loop syntax does not prove equal multiplicity when the
// runtime upper bound also depends on the launch node.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unresolved_node_dependent_rendezvous_count(%runtime: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %upper = arith.addi %runtime, %core_x : index
    scf.for %iteration = %c0 to %upper step %c1 {
      ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        // expected-error @below {{cannot prove a one-to-one synchronization schedule on PipeNet net_0 for receiver core_x=1, core_y=0; receiver post and send occurrences do not have matching proven execution counts and conditions}}
        %send = ttl.copy %send_cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
      ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %recv_reserve = ttl.cb_reserve %recv_cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        // expected-note @below {{matching receiver post occurrence is here}}
        %recv = ttl.copy %pipe, %recv_reserve
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> !ttl.receive_request
        ttl.wait %recv : !ttl.receive_request
      }
    }
    func.return
  }
}

// -----

// A loop-IV condition restricts the send to the first iteration while the
// receiver posts on both iterations.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @loop_conditional_rendezvous_count()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    scf.for %iteration = %c0 to %c2 step %c1 {
      %is_first = arith.cmpi eq, %iteration, %c0 : index
      scf.if %is_first {
        ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
          // expected-error @below {{cannot prove a one-to-one synchronization schedule on PipeNet net_0 for receiver core_x=1, core_y=0; receiver post and send occurrences do not have matching proven execution counts and conditions}}
          %send = ttl.copy %send_cb, %pipe
              : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
                 !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
              -> !ttl.transfer_handle<write>
          ttl.wait %send : !ttl.transfer_handle<write>
        }
      }
      ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %recv_reserve = ttl.cb_reserve %recv_cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        // expected-note @below {{matching receiver post occurrence is here}}
        %recv = ttl.copy %pipe, %recv_reserve
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> !ttl.receive_request
        ttl.wait %recv : !ttl.receive_request
      }
    }
    func.return
  }
}

// -----

// A send can be separated from the matching receiver post by other pipe
// events. The verifier should still report the real protocol violation instead
// of the intermediate program-order edge.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @loopback_send_before_receive_post_with_intervening_send() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %loopback_pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        {pipeNetName = "loopback_net"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %other_pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
        {pipeNetName = "other_net"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %loopback_pipe : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      // expected-error @below {{pipe send occurs before the receiver posts a dataflow buffer reservation on PipeNet loopback_net}}
      // expected-note @below {{this send waits for each destination to post `ttl.copy(pipe, dst)`}}
      // expected-note @below {{move `ttl.copy(pipe, dst)` before the dependent send, or place send and receive in separate data-movement threads}}
      %send = ttl.copy %send_cb, %loopback_pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      %other_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %other_recv = ttl.copy %other_pipe, %other_reserve
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      %other_send = ttl.copy %send_cb, %other_pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>)
          -> !ttl.transfer_handle<write>
      ttl.wait %other_send : !ttl.transfer_handle<write>
      ttl.wait %other_recv : !ttl.receive_request
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv_view = ttl.attach_cb %recv_reserve, %recv_cb
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-note @below {{this receiver post is ordered after the send in the same data-movement thread}}
      %recv = ttl.copy %loopback_pipe, %recv_view
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      ttl.wait %recv : !ttl.receive_request
    }
    func.return
  }
}

// -----

// A same-thread loopback send before the receive copy creates a cycle: the
// send waits for the receiver post, but program order executes that post after
// the send.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @loopback_send_before_receive_post() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        {pipeNetName = "net"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      // expected-error @below {{pipe send occurs before the receiver posts a dataflow buffer reservation on PipeNet net}}
      // expected-note @below {{this send waits for each destination to post `ttl.copy(pipe, dst)`}}
      // expected-note @below {{move `ttl.copy(pipe, dst)` before the dependent send, or place send and receive in separate data-movement threads}}
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv_view = ttl.attach_cb %recv_reserve, %recv_cb
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-note @below {{this receiver post is ordered after the send in the same data-movement thread}}
      %recv = ttl.copy %pipe, %recv_view
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      ttl.wait %recv : !ttl.receive_request
    }
    func.return
  }
}

// -----

// A correspondence error in one PipeNet must not suppress an independent
// wait-for-cycle diagnostic in another PipeNet.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @independent_correspondence_error()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_reserve
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      ttl.wait %recv : !ttl.receive_request
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send0 = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send0 : !ttl.transfer_handle<write>
      // expected-error @below {{PipeNet net_0 requires one static receiver post definition for each static send definition at receiver core_x=1, core_y=0; found 1 static receiver post definition(s) and 2 static send definition(s)}}
      // expected-note @below {{this send has no corresponding receiver post}}
      %send1 = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send1 : !ttl.transfer_handle<write>
    }
    func.return
  }

  func.func @independent_wait_for_cycle()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
        {pipeNetName = "loopback"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
    %send_cb = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 3, block_count = 2} {dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1> {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv_view = ttl.attach_cb %recv_reserve, %recv_cb
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_view
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      // expected-error @below {{receive wait occurs before the send that completes it on PipeNet loopback}}
      // expected-note @below {{this wait blocks until the sender transfers into the posted destination dataflow buffer slot}}
      // expected-note @below {{move the receive wait after the send, or place send and receive in separate data-movement threads}}
      ttl.wait %recv : !ttl.receive_request
      // expected-note @below {{this send is ordered after the wait in the same data-movement thread}}
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Repeated helper calls can create exponentially many pipe events. Schedule
// construction must diagnose its bound instead of exhausting memory or the
// cycle checker's call stack.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func private @expansion_leaf(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    // expected-error @below {{cannot verify PipeNet synchronization because the schedule contains more than 4096 pipe events at core_x=0, core_y=0 after expanding helper calls}}
    %send = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    func.return
  }

  func.func private @expansion_level_1(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    func.call @expansion_leaf(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_leaf(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_leaf(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_leaf(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_leaf(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_leaf(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_leaf(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_leaf(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.return
  }

  func.func private @expansion_level_2(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    func.call @expansion_level_1(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_1(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_1(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_1(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_1(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_1(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_1(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_1(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.return
  }

  func.func private @expansion_level_3(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    func.call @expansion_level_2(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_2(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_2(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_2(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_2(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_2(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_2(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_2(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.return
  }

  func.func private @expansion_level_4(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    func.call @expansion_level_3(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_3(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_3(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_3(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_3(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_3(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_3(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_3(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.return
  }

  func.func private @expansion_level_5(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    func.call @expansion_level_4(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_4(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_4(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_4(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_4(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_4(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_4(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_4(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.return
  }

  func.func @bounded_schedule_expansion()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      func.call @expansion_level_5(%send_cb, %pipe)
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    }
    func.return
  }
}

// -----

// A wait depends on the send associated with its exact receive token, not the
// next send in static order.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @wait_on_second_receive_before_second_send()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reserve0 = ttl.cb_reserve %recv_cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %receive0 = ttl.copy %pipe, %reserve0
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.receive_request
    %send0 = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send0 : !ttl.transfer_handle<write>
    %reserve1 = ttl.cb_reserve %recv_cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %receive1 = ttl.copy %pipe, %reserve1
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.receive_request
    // expected-error @below {{receive wait occurs before the send that completes it on PipeNet net_0}}
    // expected-note @below {{this wait blocks until the sender transfers into the posted destination dataflow buffer slot}}
    // expected-note @below {{move the receive wait after the send, or place send and receive in separate data-movement threads}}
    ttl.wait %receive1 : !ttl.receive_request
    // expected-note @below {{this send is ordered after the wait in the same data-movement thread}}
    %send1 = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send1 : !ttl.transfer_handle<write>
    func.return
  }
}

// -----

// Equal total counts do not make receiver-published addressing safe when all
// posts execute before any send. The second iteration overwrites the first
// posted address.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @repeated_receive_ahead()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      scf.for %iteration = %c0 to %c2 step %c1 {
        %reserve = ttl.cb_reserve %recv_cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        // expected-error @below {{cannot prove that each repeated receiver post is consumed before the next post on PipeNet net_0 at core_x=1, core_y=0}}
        %receive = ttl.copy %pipe, %reserve
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> !ttl.receive_request
      }
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      scf.for %iteration = %c0 to %c2 step %c1 {
        %send = ttl.copy %send_cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
    }
    func.return
  }
}

// -----

// Equal static counts do not make two receiver posts safe when the first send
// is not proven complete before the second address publication.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @two_static_posts_before_sends_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %reserve0 = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-note @below {{the preceding receiver post is not proven consumed before this post}}
      %receive0 = ttl.copy %pipe, %reserve0
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      %reserve1 = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-error @below {{receiver post may overwrite an outstanding posted address on PipeNet net_0 at core_x=1, core_y=0}}
      %receive1 = ttl.copy %pipe, %reserve1
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
    }
    func.return
  }

  func.func @two_static_posts_before_sends_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send0 = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send0 : !ttl.transfer_handle<write>
      %send1 = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send1 : !ttl.transfer_handle<write>
    }
    func.return
  }
}
