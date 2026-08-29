// Verifies diagnostics for ready-receive candidates without one logical
// receive identity.

// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(convert-ttl-to-ttkernel)'

// Every request must originate from a pipe receive copy.
func.func @request_requires_pipe_receive_origin(
    %request: !ttl.receive_request) {
  %start = arith.constant 0 : index
  // expected-error @below {{requires every request origin to be a pipe receive ttl.copy}}
  %ready = ttl.wait_any %request start %start
      : (!ttl.receive_request, index) -> !ttl.ready_receive
  func.return
}

// -----

// Distinct candidates cannot refer to the same receive copy.
func.func @candidates_require_disjoint_receive_origins() {
  %landing = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %dst = ttl.cb_reserve %landing
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %request = ttl.copy %pipe, %dst
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  %same_request = builtin.unrealized_conversion_cast %request
      : !ttl.receive_request to !ttl.receive_request
  %start = arith.constant 0 : index
  // expected-error @below {{requires request values with disjoint pipe receive origins}}
  %ready = ttl.wait_any %request, %same_request start %start
      : (!ttl.receive_request, !ttl.receive_request, index)
      -> !ttl.ready_receive
  func.return
}

// -----

// A request merged from different PipeNets has no single logical tag.
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @merged_pipenets(%condition: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %landing = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
    %request = scf.if %condition -> (!ttl.receive_request) {
      %dst = ttl.cb_reserve %landing
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %then_request = ttl.copy %pipe0, %dst
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      scf.yield %then_request : !ttl.receive_request
    } else {
      %dst = ttl.cb_reserve %landing
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %else_request = ttl.copy %pipe1, %dst
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      scf.yield %else_request : !ttl.receive_request
    }
    %start = arith.constant 0 : index
    // expected-error @below {{requires each request's origins to belong to one PipeNet}}
    %ready = ttl.wait_any %request start %start
        : (!ttl.receive_request, index) -> !ttl.ready_receive
    func.return
  }
}

// -----

// Alternate origins for one candidate must reserve the same destination DFB
// stream.
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @merged_destination_streams(%condition: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %landing0 = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %landing1 = ttl.bind_cb {cb_index = 2, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %request = scf.if %condition -> (!ttl.receive_request) {
      %dst = ttl.cb_reserve %landing0
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %then_request = ttl.copy %pipe, %dst
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      %send = ttl.copy %source, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      scf.yield %then_request : !ttl.receive_request
    } else {
      %dst = ttl.cb_reserve %landing1
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %else_request = ttl.copy %pipe, %dst
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      %send = ttl.copy %source, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      scf.yield %else_request : !ttl.receive_request
    }
    %start = arith.constant 0 : index
    // expected-error @below {{requires each candidate's possible posts to use one logical receive channel and destination DFB stream}}
    %ready = ttl.wait_any %request start %start
        : (!ttl.receive_request, index) -> !ttl.ready_receive
    func.return
  }
}
