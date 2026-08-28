// Verify that automatic DFB synchronization rejects external release effects
// that cannot be relocated without deleting the external call.
// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(ttl-insert-cb-sync))' --verify-diagnostics

// A nested external push cannot satisfy an entry-block reserve.
module {
  func.func @nested_external_push(%condition: i1) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reserved = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.if %condition {
      // expected-error @below {{external DFB push effect must be in the same block as its acquisition}}
      ttl.opaque_call "publish" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<push, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    return
  }
}

// -----

// A nested external pop cannot satisfy an entry-block wait.
module {
  func.func @nested_external_pop(%condition: i1) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %waited = ttl.cb_wait %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.if %condition {
      // expected-error @below {{external DFB pop effect must be in the same block as its acquisition}}
      ttl.opaque_call "release" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    return
  }
}

// -----

// A same-block release cannot precede a tensor use of the acquired slot.
module {
  func.func @same_block_release_before_tensor_use()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %output = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %waited = ttl.cb_wait %input : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %input_block = ttl.attach_cb %waited, %input : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reserved = ttl.cb_reserve %output : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{dataflow buffer push must follow all uses owned by its acquisition}}
    ttl.cb_push %output : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.store %input_block, %reserved : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %input : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// A guarded local release cannot precede another use of the acquired slot.
module {
  func.func @guarded_local_release_before_local_use(
      %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %condition: i1) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %view = scf.if %condition -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
      %reserved = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-error @below {{guarded local dataflow buffer push must follow all uses in its acquiring region}}
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      ttl.store %arg0, %reserved : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield %reserved : tensor<1x1x!ttcore.tile<32x32, bf16>>
    } else {
      %inactive = "builtin.unrealized_conversion_cast"() {ttl.inactive_guarded_dfb} : () -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield %inactive : tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    return
  }
}

// -----

// A nested region in the acquiring block may capture the acquired slot.
module {
  func.func @guarded_local_release_before_nested_local_use(
      %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %condition: i1) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %view = scf.if %condition -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
      %reserved = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-error @below {{guarded local dataflow buffer push must follow all uses in its acquiring region}}
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      scf.if %condition {
        ttl.store %arg0, %reserved : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
      scf.yield %reserved : tensor<1x1x!ttcore.tile<32x32, bf16>>
    } else {
      %inactive = "builtin.unrealized_conversion_cast"() {ttl.inactive_guarded_dfb} : () -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield %inactive : tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    return
  }
}

// -----

// External releases cannot be moved to a sibling guarded region when the
// acquired slot is used after the acquiring region.
module {
  func.func @guarded_local_external_release(
      %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %condition: i1) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %view = scf.if %condition -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
      %reserved = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-error @below {{external dataflow buffer push effect cannot be relocated out of a guarded acquisition region}}
      ttl.opaque_call "publish" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<push, 0, 1>] () {header = "effects.hpp"} : () -> ()
      scf.yield %reserved : tensor<1x1x!ttcore.tile<32x32, bf16>>
    } else {
      %inactive = "builtin.unrealized_conversion_cast"() {ttl.inactive_guarded_dfb} : () -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield %inactive : tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    scf.if %condition {
      %sum = ttl.add %view, %arg0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    return
  }
}

// -----

// A guarded acquire result cannot be used when the acquire condition may be
// false.
module {
  func.func @guarded_wait_unguarded_escape(
      %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %condition: i1) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %view = scf.if %condition -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
      %waited = ttl.cb_wait %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield %waited : tensor<1x1x!ttcore.tile<32x32, bf16>>
    } else {
      %inactive = "builtin.unrealized_conversion_cast"() {ttl.inactive_guarded_dfb} : () -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield %inactive : tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    // expected-error @below {{conditional dataflow buffer slot use must be under the acquiring condition}}
    %sum = ttl.add %view, %arg0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// A guarded external release must execute under the acquire condition.
module {
  func.func @guarded_wait_released_under_negated_condition(
      %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %condition: i1) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %true = arith.constant true
    %not_condition = arith.xori %condition, %true : i1
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %view = scf.if %condition -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
      %waited = ttl.cb_wait %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield %waited : tensor<1x1x!ttcore.tile<32x32, bf16>>
    } else {
      %inactive = "builtin.unrealized_conversion_cast"() {ttl.inactive_guarded_dfb} : () -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield %inactive : tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    scf.if %condition {
      %sum = ttl.add %view, %arg0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    scf.if %not_condition {
      // expected-error @below {{conditional dataflow buffer pop must execute under the acquiring condition}}
      ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    return
  }
}

// -----

// A guarded external release must follow same-condition uses of the acquired
// slot.
module {
  func.func @guarded_reserve_released_before_same_condition_use(
      %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %condition: i1) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %view = scf.if %condition -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
      %reserved = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield %reserved : tensor<1x1x!ttcore.tile<32x32, bf16>>
    } else {
      %inactive = "builtin.unrealized_conversion_cast"() {ttl.inactive_guarded_dfb} : () -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield %inactive : tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    scf.if %condition {
      // expected-error @below {{conditional dataflow buffer push must follow all uses under the acquiring condition}}
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    scf.if %condition {
      ttl.store %arg0, %view : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    return
  }
}

// -----

// Wait-any proves one candidate complete, not every candidate.
func.func @unguarded_wait_any_push()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %landing0 = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %landing1 = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
  %block0 = ttl.cb_reserve %landing0
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %block1 = ttl.cb_reserve %landing1
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %request0 = ttl.copy %pipe0, %block0
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>) -> !ttl.receive_request
  %request1 = ttl.copy %pipe1, %block1
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>) -> !ttl.receive_request
  %start = arith.constant 0 : index
  %ready = ttl.wait_any %request0, %request1 start %start
      : (!ttl.receive_request, !ttl.receive_request, index)
      -> !ttl.ready_receive
  // expected-error @below {{publishes a wait-any receive reservation without proving that candidate complete}}
  ttl.cb_push %landing0
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

// Every wait-any receive reservation requires an explicit publication.
func.func @unpublished_wait_any_reservations()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %landing0 = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %landing1 = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
  // expected-error @below {{wait-any receive reservation is never published}}
  %block0 = ttl.cb_reserve %landing0
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %request0 = ttl.copy %pipe0, %block0
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>) -> !ttl.receive_request
  %block1 = ttl.cb_reserve %landing1
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %request1 = ttl.copy %pipe1, %block1
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>) -> !ttl.receive_request
  %start = arith.constant 0 : index
  %ready = ttl.wait_any %request0, %request1 start %start
      : (!ttl.receive_request, !ttl.receive_request, index)
      -> !ttl.ready_receive
  func.return
}

// -----

// Selection may consume candidates out of reservation order, so candidates
// published according to the selected index require separate dataflow buffer
// streams.
func.func @selected_publication_on_shared_stream()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %landing = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
  %block0 = ttl.cb_reserve %landing
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %request0 = ttl.copy %pipe0, %block0
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>) -> !ttl.receive_request
  %block1 = ttl.cb_reserve %landing
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %request1 = ttl.copy %pipe1, %block1
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>) -> !ttl.receive_request
  %start = arith.constant 0 : index
  %ready = ttl.wait_any %request0, %request1 start %start
      : (!ttl.receive_request, !ttl.receive_request, index)
      -> !ttl.ready_receive
  %selected = ttl.ready_receive_index %ready : !ttl.ready_receive
  %zero = arith.constant 0 : index
  %selected0 = arith.cmpi eq, %selected, %zero : index
  scf.if %selected0 {
    ttl.wait %request0 : !ttl.receive_request
    // expected-error @below {{wait-any candidates published according to selection must use separate destination dataflow buffer streams}}
    ttl.cb_push %landing
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  }
  func.return
}
