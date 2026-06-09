// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -ttl-verify-dfb-spsc

// Summary: Negative tests for per-launch-node DFB SPSC verification.

// Two consumer threads overlap on core (0, 0), so the DFB is not SPSC.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @consumer_all_nodes() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{dataflow buffer cb_index=0 has multiple consumer threads active on the same launched node}}
    // expected-note @below {{example overlapping node: core_x=0, core_y=0}}
    // expected-note @below {{tt-metal CBs are single-producer single-consumer; allocate one DFB per consumer}}
    %view = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @consumer_x0() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %is_x0 = arith.cmpi eq, %core_x, %zero : index
    scf.if %is_x0 {
      // expected-note @below {{also waited on here}}
      %view = ttl.cb_wait %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    func.return
  }
}

// -----

// Two producer threads overlap on core (1, 0), so the DFB is not SPSC.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @producer_all_nodes() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %cb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{dataflow buffer cb_index=1 has multiple producer threads active on the same launched node}}
    // expected-note @below {{example overlapping node: core_x=1, core_y=0}}
    // expected-note @below {{tt-metal CBs are single-producer single-consumer; allocate one DFB per producer}}
    %slot = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @producer_x1() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %one = arith.constant 1 : index
    %is_x1 = arith.cmpi eq, %core_x, %one : index
    scf.if %is_x1 {
      // expected-note @below {{also reserved here}}
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
    %cb = ttl.bind_cb {cb_index = 2, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %scaled = arith.muli %core_x, %runtime : index
    %zero = arith.constant 0 : index
    // expected-note @below {{this expression is not statically analyzable}}
    %cond = arith.cmpi eq, %scaled, %zero : index
    scf.if %cond {
      %is_x0 = arith.cmpi eq, %core_x, %zero : index
      scf.if %is_x0 {
        // expected-error @below {{dataflow buffer cb_index=2 has multiple consumer threads, but SPSC could not be statically proven}}
        // expected-note @below {{tt-metal CBs are single-producer single-consumer; allocate one DFB per consumer}}
        %view = ttl.cb_wait %cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      }
    }
    func.return
  }

  func.func @other_consumer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 2, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-note @below {{also waited on here}}
    %view = ttl.cb_wait %cb
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
    %cb = ttl.bind_cb {cb_index = 3, block_count = 2}
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
    %cb = ttl.bind_cb {cb_index = 4, block_count = 2}
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
    %cb = ttl.bind_cb {cb_index = 5, block_count = 2}
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
