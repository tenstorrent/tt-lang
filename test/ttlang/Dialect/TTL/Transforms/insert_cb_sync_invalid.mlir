// Verify that automatic DFB synchronization rejects external release effects
// that cannot be relocated without deleting the external call.
// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(ttl-insert-cb-sync))' --verify-diagnostics

// A later acquisition does not make an earlier slot valid after publication.
func.func @guarded_slot_used_after_publication_and_later_acquire(
    %value: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %condition = arith.constant true
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  scf.if %condition {
    %first_reserved = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %value, %first_reserved : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{guarded local dataflow buffer push must follow all uses in its acquiring region}}
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    scf.if %condition {
      %second_reserved = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.opaque_call "fill_later_slot" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) () {header = "fill.hpp"} : () -> ()
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    ttl.store %value, %first_reserved : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  func.return
}

// -----

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

// A later acquisition does not take ownership of tensor SSA values returned
// by an earlier acquisition.
module {
  func.func @guarded_reserve_released_before_ssa_use_across_later_acquire(
      %value: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %condition: i1) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    scf.if %condition {
      %first_reserved = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-error @below {{guarded local dataflow buffer push must follow all uses in its acquiring region}}
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      scf.if %condition {
        %second_reserved = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.opaque_call "fill_later_slot" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) () {header = "fill.hpp"} : () -> ()
        ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      }
      ttl.store %value, %first_reserved : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
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

// -----

// A waited block released after a loop that waits on the same DFB would alias
// the loop's waits; the release after the loop marks the program invalid.

func.func @wait_held_across_nested_wait()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %in = ttl.bind_cb{cb_index = 0, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %out = ttl.bind_cb{cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  // expected-error @below {{dataflow buffer block is still acquired when a nested region acquires the same buffer again; release it before that region (pop it) or acquire the block inside the region}}
  %w0 = ttl.cb_wait %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b0 = ttl.attach_cb %w0, %in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %reserve = ttl.cb_reserve %out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %b0, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.for %iv = %c0 to %c3 step %c1 {
    // expected-note @below {{the buffer is acquired again here}}
    %w = ttl.cb_wait %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.attach_cb %w, %in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %b, %reserve {accumulate} : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  }
  ttl.cb_push %out : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  func.return
}

// -----

// A data-movement kernel addresses the DFB through one write pointer, so a
// second reserve while the first block is unused and unreleased is rejected:
// the copies that follow would all address the second block.

func.func @dm_two_open_reserves(%arg0: tensor<2x8x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cb = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %first = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{a data-movement kernel cannot hold two acquired blocks of one dataflow buffer; the earlier block has no use before this acquisition, so use and push it before this acquisition or drop it}}
  %second = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %s0 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<2x8x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>>
  %x0 = ttl.copy %s0, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %x0 : !ttl.transfer_handle<read>
  %s1 = ttl.tensor_slice %arg0[%c0, %c1] : tensor<2x8x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>>
  %x1 = ttl.copy %s1, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %x1 : !ttl.transfer_handle<read>
  ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// A reserved block held across a loop that reserves the same DFB again.

func.func @reserve_held_across_nested_reserve()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %out = ttl.bind_cb{cb_index = 0, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %in = ttl.bind_cb{cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  // expected-error @below {{dataflow buffer block is still acquired when a nested region acquires the same buffer again; release it before that region (push it) or acquire the block inside the region}}
  %r0 = ttl.cb_reserve %out : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %w0 = ttl.cb_wait %in : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b0 = ttl.attach_cb %w0, %in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %b0, %r0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %in : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  scf.for %iv = %c0 to %c3 step %c1 {
    // expected-note @below {{the buffer is acquired again here}}
    %r = ttl.cb_reserve %out : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %w = ttl.cb_wait %in : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.attach_cb %w, %in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %b, %r : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %in : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_push %out : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  }
  ttl.cb_push %out : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  func.return
}

// -----

// A use inside the region before the nested wait cannot be followed by a
// release after the region: the release would land after the aliasing wait.

func.func @use_in_region_release_after_region(%condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %in = ttl.bind_cb{cb_index = 0, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %out = ttl.bind_cb{cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  // expected-error @below {{dataflow buffer block is still acquired when a nested region acquires the same buffer again; release it before that region (pop it) or acquire the block inside the region}}
  %w0 = ttl.cb_wait %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b0 = ttl.attach_cb %w0, %in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.if %condition {
    ttl.store %b0, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-note @below {{the buffer is acquired again here}}
    %w = ttl.cb_wait %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.attach_cb %w, %in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %b, %reserve {accumulate} : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  }
  ttl.cb_push %out : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  func.return
}

// -----

// A release inside a loop body before the nested wait frees the block for
// the first iteration only.

func.func @release_inside_loop_before_nested_wait()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %in = ttl.bind_cb{cb_index = 0, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %out = ttl.bind_cb{cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  // expected-error @below {{dataflow buffer block is still acquired when a nested region acquires the same buffer again; release it before that region (pop it) or acquire the block inside the region}}
  %w0 = ttl.cb_wait %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b0 = ttl.attach_cb %w0, %in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %b0, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  scf.for %iv = %c0 to %c3 step %c1 {
    %first = arith.cmpi eq, %iv, %c0 : index
    scf.if %first {
      ttl.cb_pop %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    }
    // expected-note @below {{the buffer is acquired again here}}
    %w = ttl.cb_wait %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.attach_cb %w, %in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %b, %reserve {accumulate} : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  }
  ttl.cb_push %out : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// A guarded acquisition is checked like an unguarded one.

func.func @guarded_wait_held_across_nested_wait(%condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %in = ttl.bind_cb{cb_index = 0, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %out = ttl.bind_cb{cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve = ttl.cb_reserve %out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %held = scf.if %condition -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
    // expected-error @below {{dataflow buffer block is still acquired when a nested region acquires the same buffer again; release it before that region (pop it) or acquire the block inside the region}}
    %w0 = ttl.cb_wait %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b0 = ttl.attach_cb %w0, %in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %b0, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %b0 : tensor<1x1x!ttcore.tile<32x32, bf16>>
  } else {
    %inactive = "builtin.unrealized_conversion_cast"() {ttl.inactive_guarded_dfb} : () -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %inactive : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  scf.for %iv = %c0 to %c3 step %c1 {
    // expected-note @below {{the buffer is acquired again here}}
    %w = ttl.cb_wait %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.attach_cb %w, %in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %b, %reserve {accumulate} : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  }
  ttl.cb_push %out : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  scf.if %condition {
    ttl.cb_pop %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  }
  func.return
}

// -----

// The branch that acquires the buffer again has no release of the held block,
// and the other branch uses the block, so no single placement serves both.

func.func @use_in_other_branch_without_release_on_acquiring_path(
    %arg0: tensor<2x8x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>>, %condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cb = ttl.bind_cb{cb_index = 0, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  // expected-error @below {{dataflow buffer block is still acquired when a nested region acquires the same buffer again; release it before that region (pop it) or on every path through the region}}
  %w0 = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.if %condition {
    // expected-note @below {{the buffer is acquired again here}}
    %w1 = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %s1 = ttl.tensor_slice %arg0[%c0, %c1] : tensor<2x8x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>>
    %x1 = ttl.copy %cb, %s1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>, tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>>) -> !ttl.transfer_handle<write>
    ttl.wait %x1 : !ttl.transfer_handle<write>
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  } else {
    %value = ttl.raw_element_read %w0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> bf16
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  }
  func.return
}

// -----

// A release in a sibling nested region does not execute on every path to the
// nested wait.

func.func @release_in_sibling_nested_region(%condition: i1, %inner: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %in = ttl.bind_cb{cb_index = 0, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %out = ttl.bind_cb{cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  // expected-error @below {{dataflow buffer block is still acquired when a nested region acquires the same buffer again; release it before that region (pop it) or acquire the block inside the region}}
  %w0 = ttl.cb_wait %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b0 = ttl.attach_cb %w0, %in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %b0, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.if %condition {
    scf.if %inner {
      ttl.cb_pop %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    }
    // expected-note @below {{the buffer is acquired again here}}
    %w = ttl.cb_wait %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.attach_cb %w, %in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %b, %reserve {accumulate} : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  }
  ttl.cb_push %out : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// A release after the nested wait inside its region releases the held block
// after the aliasing wait.

func.func @release_after_nested_wait_inside_region(%condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %in = ttl.bind_cb{cb_index = 0, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %out = ttl.bind_cb{cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  // expected-error @below {{dataflow buffer block is still acquired when a nested region acquires the same buffer again; release it before that region (pop it) or acquire the block inside the region}}
  %w0 = ttl.cb_wait %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b0 = ttl.attach_cb %w0, %in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %b0, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.if %condition {
    // expected-note @below {{the buffer is acquired again here}}
    %w = ttl.cb_wait %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.attach_cb %w, %in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %b, %reserve {accumulate} : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    ttl.cb_pop %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  }
  ttl.cb_push %out : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// An index switch whose acquiring case lacks a release while another case
// uses the held block.

func.func @index_switch_release_missing_in_acquiring_case(%selector: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %in = ttl.bind_cb{cb_index = 0, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %out = ttl.bind_cb{cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  // expected-error @below {{dataflow buffer block is still acquired when a nested region acquires the same buffer again; release it before that region (pop it) or on every path through the region}}
  %w0 = ttl.cb_wait %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b0 = ttl.attach_cb %w0, %in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.index_switch %selector
  case 0 {
    // expected-note @below {{the buffer is acquired again here}}
    %w = ttl.cb_wait %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %b = ttl.attach_cb %w, %in : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %b, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    scf.yield
  }
  default {
    ttl.store %b0, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  }
  ttl.cb_push %out : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// A data-movement kernel that uses the first block, reserves again, and
// releases both blocks afterwards places the first release after the second
// reserve.

func.func @dm_release_after_second_reserve(
    %arg0: tensor<2x8x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cb = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  scf.for %iv = %c0 to %c2 step %c1 {
    %first = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %s0 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<2x8x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>>
    %x0 = ttl.copy %s0, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %x0 : !ttl.transfer_handle<read>
    // expected-error @below {{a data-movement kernel cannot hold two acquired blocks of one dataflow buffer; the earlier block is released after this acquisition, so push it before this acquisition}}
    %second = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %s1 = ttl.tensor_slice %arg0[%c0, %c1] : tensor<2x8x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>>
    %x1 = ttl.copy %s1, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %x1 : !ttl.transfer_handle<read>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  }
  func.return
}

// -----

// Two open reserves inside a node-conditional region of a data-movement
// kernel are rejected like two open reserves at the top level.

func.func @dm_two_open_reserves_in_branch(
    %arg0: tensor<2x8x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>>, %condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cb = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  scf.if %condition {
    %first = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{a data-movement kernel cannot hold two acquired blocks of one dataflow buffer; the earlier block has no use before this acquisition, so use and push it before this acquisition or drop it}}
    %second = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %s0 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<2x8x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>>
    %x0 = ttl.copy %s0, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %x0 : !ttl.transfer_handle<read>
    %s1 = ttl.tensor_slice %arg0[%c0, %c1] : tensor<2x8x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>> -> tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>>
    %x1 = ttl.copy %s1, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [64, 256], element_type = !ttcore.tile<32x32, bf16>, buffer = system_memory, grid = [1, 1], memory = interleaved>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %x1 : !ttl.transfer_handle<read>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  }
  func.return
}
