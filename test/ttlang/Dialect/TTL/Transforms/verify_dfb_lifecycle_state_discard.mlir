// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices,ttl-verify-dfb-lifecycle)'

// Summary: A synchronized reset or state-discarding reconfiguration restores
// only the DFB interfaces the runtime resets. The lifecycle verifier takes
// reset targets from the lowered reset masks and reconfiguration targets from
// the descriptors the finalized plan installs at each boundary, so a DFB that
// the plan keeps across a boundary keeps its queue state in verification too.

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary0 = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer], discard_dfb_state = true>
#boundary1 = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer], discard_dfb_state = true>

// A token published before a state-discarding reconfiguration and consumed
// after it is live across the boundary, so the plan keeps its descriptor
// there. Its sequence continues across the boundary and is accepted; treating
// the boundary as a reset of every DFB would make the consumer's wait precede
// every publication in its interval.
module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %iteration = %c0 to %c2 step %c1 {
      ttl.dfb_reconfiguration #boundary0
      ttl.dfb_reconfiguration #boundary1
    }
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %token = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %iteration = %c0 to %c2 step %c1 {
      ttl.opaque_call "publish_token" dfb_dependencies(
          %token : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
          dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                       #ttl.dfb_protocol_effect<push, 0, 1>]
          () {header = "token.hpp"} : () -> ()
      ttl.dfb_reconfiguration #boundary0
      ttl.opaque_call "consume_token" dfb_dependencies(
          %token : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
          dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>,
                       #ttl.dfb_protocol_effect<pop, 0, 1>]
          () {header = "token.hpp"} : () -> ()
      ttl.dfb_reconfiguration #boundary1
    }
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %iteration = %c0 to %c2 step %c1 {
      ttl.dfb_reconfiguration #boundary0
      ttl.dfb_reconfiguration #boundary1
    }
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary0 = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer], discard_dfb_state = true>
#boundary1 = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer], discard_dfb_state = true>

// An external producer declares its protocol effects, and the writer waits
// for the published page without popping it. The plan bounds the lifecycle
// and installs a new descriptor at a state-discarding boundary in every
// iteration, which releases the waited page.
module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %partial = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %iteration = %c0 to %c4 step %c1 {
      ttl.opaque_call "produce_partial" dfb_dependencies(
          %partial : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
          dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                       #ttl.dfb_protocol_effect<push, 0, 1>]
          () {header = "partial.hpp"} : () -> ()
      ttl.dfb_reconfiguration #boundary0
      ttl.dfb_reconfiguration #boundary1
    }
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %iteration = %c0 to %c4 step %c1 {
      ttl.dfb_reconfiguration #boundary0
      ttl.dfb_reconfiguration #boundary1
    }
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    %partial = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %iteration = %c0 to %c4 step %c1 {
      ttl.opaque_call "read_partial" dfb_dependencies(
          %partial : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
          dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>]
          () {header = "partial.hpp"} : () -> ()
      ttl.dfb_reconfiguration #boundary0
      ttl.dfb_reconfiguration #boundary1
    }
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary0 = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer], discard_dfb_state = true>
#boundary1 = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer], discard_dfb_state = true>

// The same program, except that the external producer declares no protocol
// effects and the compute kernel announces its role with a push under a
// runtime condition the compiler cannot resolve. The plan cannot bound the
// lifecycle, so it installs a new descriptor at no boundary and the waited
// page is never released. The lifecycle verifier cannot check this node
// because of the undeclared external actions, and warns instead.
module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    // expected-note @+1 {{dataflow buffer declared here}}
    %partial = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %zero = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %iteration = %c0 to %c4 step %c1 {
      // expected-note @+1 {{this external call may perform protocol actions on the DFB that it does not declare}}
      ttl.opaque_call "produce_partial" dfb_dependencies(
          %partial : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
          () {header = "partial.hpp"} : () -> ()
      %flag = ttl.opaque_call "runtime_flag" () {header = "partial.hpp"} : () -> i64
      %announce = arith.cmpi ne, %flag, %zero : i64
      scf.if %announce {
        %marker = ttl.cb_reserve %partial : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_push %partial : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      }
      ttl.dfb_reconfiguration #boundary0
      ttl.dfb_reconfiguration #boundary1
    }
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %iteration = %c0 to %c4 step %c1 {
      ttl.dfb_reconfiguration #boundary0
      ttl.dfb_reconfiguration #boundary1
    }
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    %partial = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %iteration = %c0 to %c4 step %c1 {
      // expected-warning @+3 {{logical DFB 0 is never popped on core_x=0, core_y=0, but its producer can push 4 block(s) into capacity 1 during the launch}}
      // expected-note @+2 {{published blocks stay in the DFB until a pop, so the producer blocks once the DFB is full}}
      // expected-note @+1 {{a reconfiguration restores a DFB only where the reconfiguration plan installs a new descriptor for it, which requires a bounded lifecycle; declare the DFB effects of external calls that access it, or pop the published blocks}}
      ttl.opaque_call "read_partial" dfb_dependencies(
          %partial : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
          dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>]
          () {header = "partial.hpp"} : () -> ()
      ttl.dfb_reconfiguration #boundary0
      ttl.dfb_reconfiguration #boundary1
    }
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary0 = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer], discard_dfb_state = false>

// A wait held across a reconfiguration that keeps the DFB is closed by an
// external pop after the boundary. The pop closes the wait still open from
// before the boundary instead of acquiring another block.
module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    ttl.dfb_reconfiguration #boundary0
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %reserved = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %waited = ttl.cb_wait %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.dfb_reconfiguration #boundary0
    ttl.opaque_call "consume" dfb_dependencies(
        %dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        dfb_effects [#ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "consume.hpp"} : () -> ()
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    ttl.dfb_reconfiguration #boundary0
    return
  }
}
