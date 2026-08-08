// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices,ttl-verify-dfb-spsc)'

// Summary: Negative tests for per-launch-node DFB SPSC verification.

// Two consumer threads overlap on core (0, 0), so the DFB is not SPSC.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @consumer_all_nodes() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{logical DFB 0 has multiple consumer kernels active on the same launched node}}
    // expected-note @below {{example overlapping node: core_x=0, core_y=0}}
    // expected-note @below {{tt-metal CBs are single-producer single-consumer; allocate one DFB per consumer}}
    %view = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @consumer_x0() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %is_x0 = arith.cmpi eq, %core_x, %zero : index
    scf.if %is_x0 {
      // expected-note @below {{also performed a consumer action here}}
      %view = ttl.cb_wait %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    func.return
  }
}

// -----

// Hidden reserve effects in two overlapping kernels violate SPSC.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @first_hidden_producer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 41 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{logical DFB 41 has multiple producer kernels active on the same launched node}}
    // expected-note @below {{example overlapping node: core_x=0, core_y=0}}
    // expected-note @below {{tt-metal CBs are single-producer single-consumer; allocate one DFB per producer}}
    ttl.opaque_call "produce_a" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>] () {header = "effects.hpp"} : () -> ()
    func.return
  }

  func.func @second_hidden_producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 41 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-note @below {{also performed a producer action here}}
    ttl.opaque_call "produce_b" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>] () {header = "effects.hpp"} : () -> ()
    func.return
  }
}

// -----

// Hidden wait effects in two overlapping kernels violate SPSC.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @first_hidden_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 42 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{logical DFB 42 has multiple consumer kernels active on the same launched node}}
    // expected-note @below {{example overlapping node: core_x=0, core_y=0}}
    // expected-note @below {{tt-metal CBs are single-producer single-consumer; allocate one DFB per consumer}}
    ttl.opaque_call "consume_a" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>] () {header = "effects.hpp"} : () -> ()
    func.return
  }

  func.func @second_hidden_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 42 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-note @below {{also performed a consumer action here}}
    ttl.opaque_call "consume_b" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>] () {header = "effects.hpp"} : () -> ()
    func.return
  }
}

// -----

// A hidden push is a producer action and cannot run in a different thread from
// the reserve on the same DFB.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @reserver() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 43 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{logical DFB 43 has multiple producer kernels active on the same launched node}}
    // expected-note @below {{example overlapping node: core_x=0, core_y=0}}
    // expected-note @below {{tt-metal CBs are single-producer single-consumer; allocate one DFB per producer}}
    %slot = ttl.cb_reserve %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @hidden_pusher() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 43 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-note @below {{also performed a producer action here}}
    ttl.opaque_call "push" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<push, 0, 1>] () {header = "effects.hpp"} : () -> ()
    func.return
  }
}

// -----

// A hidden pop is a consumer action and cannot run in a different thread from
// the wait on the same DFB.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @waiter() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 44 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{logical DFB 44 has multiple consumer kernels active on the same launched node}}
    // expected-note @below {{example overlapping node: core_x=0, core_y=0}}
    // expected-note @below {{tt-metal CBs are single-producer single-consumer; allocate one DFB per consumer}}
    %slot = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @hidden_popper() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 44 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-note @below {{also performed a consumer action here}}
    ttl.opaque_call "pop" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    func.return
  }
}

// -----

// Two producer threads overlap on core (1, 0), so the DFB is not SPSC.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @producer_all_nodes() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{logical DFB 1 has multiple producer kernels active on the same launched node}}
    // expected-note @below {{example overlapping node: core_x=1, core_y=0}}
    // expected-note @below {{tt-metal CBs are single-producer single-consumer; allocate one DFB per producer}}
    %slot = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @producer_x1() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %one = arith.constant 1 : index
    %is_x1 = arith.cmpi eq, %core_x, %one : index
    scf.if %is_x1 {
      // expected-note @below {{also performed a producer action here}}
      %slot = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    func.return
  }
}

// -----

// Unknown coord-dependent predicates are rejected when multiple consumers
// participate because the verifier cannot prove their domains are disjoint.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unknown_consumer(%runtime: index) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %scaled = arith.muli %core_x, %runtime : index
    %zero = arith.constant 0 : index
    // expected-note @below {{this expression is not statically analyzable}}
    %cond = arith.cmpi eq, %scaled, %zero : index
    scf.if %cond {
      %is_x0 = arith.cmpi eq, %core_x, %zero : index
      scf.if %is_x0 {
        // expected-error @below {{logical DFB 2 has multiple consumer kernels, but SPSC could not be statically proven}}
        // expected-note @below {{tt-metal CBs are single-producer single-consumer; allocate one DFB per consumer}}
        %view = ttl.cb_wait %cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
    }
    func.return
  }

  func.func @other_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-note @below {{also performed a consumer action here}}
    %view = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Unknown coordinate-dependent predicates are rejected when multiple producer
// kernels participate because the verifier cannot prove disjoint domains.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unknown_producer(%runtime: index) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 6 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %scaled = arith.muli %core_x, %runtime : index
    %zero = arith.constant 0 : index
    // expected-note @below {{this expression is not statically analyzable}}
    %cond = arith.cmpi eq, %scaled, %zero : index
    scf.if %cond {
      %is_x0 = arith.cmpi eq, %core_x, %zero : index
      scf.if %is_x0 {
        // expected-error @below {{logical DFB 6 has multiple producer kernels, but SPSC could not be statically proven}}
        // expected-note @below {{tt-metal CBs are single-producer single-consumer; allocate one DFB per producer}}
        %slot = ttl.cb_reserve %cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
    }
    func.return
  }

  func.func @other_producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 6 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-note @below {{also performed a producer action here}}
    %slot = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// DFB acquire verification requires a launch grid.

// expected-error @below {{ttl-verify-dfb-spsc requires a `ttl.launch_grid` module attribute}}
module {
  func.func @missing_launch_grid() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %view = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// DFB acquire verification requires a valid launch grid.

// expected-error @below {{ttl-verify-dfb-spsc requires a `ttl.launch_grid` module attribute}}
module attributes {ttl.launch_grid = [0 : i64, 1 : i64]} {
  func.func @malformed_launch_grid() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 4 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %view = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Malformed PipeNet scope metadata must make the pass fail, even when the DFB
// participant set would otherwise be accepted.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @malformed_pipenet_scope() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 5 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{has invalid PipeNet role 7}}
    ttl.pipenet_scope attributes {ttl.pipe_net_ids = [0 : i64], ttl.pipe_net_roles = [7 : i64]} {
      %view = ttl.cb_wait %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    func.return
  }
}
