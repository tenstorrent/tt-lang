// Summary: Negative tests for finalized DFB verifier preconditions.

// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -ttl-verify-dfb-spsc
// RUN: env TTL_RELAX_DFB_SPSC=1 ttlang-opt %s --split-input-file --verify-diagnostics -ttl-verify-dfb-spsc

// SPSC verification requires the allocation metadata emitted by DFB
// finalization, even when the frontend already assigned a logical ID.

// expected-error @below {{`ttl-verify-dfb-spsc` requires finalized DFB allocation metadata; run `ttl-finalize-dfb-indices` first}}
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @missing_finalization()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// A finalized DFB declaration must contain its module-wide logical ID.
// The empty allocation table is synthetic and isolates this precondition.

module attributes {
  ttl.dfb_allocations = [],
  ttl.launch_grid = [1 : i64, 1 : i64]
} {
  func.func @missing_logical_id()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-error @below {{`ttl-verify-dfb-spsc` requires every `ttl.bind_cb` to have `dfb_id` after finalization}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Relaxed domain verification still requires a launch grid.

// expected-error @below {{ttl-verify-dfb-spsc requires a `ttl.launch_grid` module attribute}}
module attributes {ttl.dfb_allocations = []} {
  func.func @missing_launch_grid()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Relaxed domain verification still requires a valid launch grid.

// expected-error @below {{ttl-verify-dfb-spsc requires a `ttl.launch_grid` module attribute}}
module attributes {
  ttl.dfb_allocations = [],
  ttl.launch_grid = [0 : i64, 1 : i64]
} {
  func.func @malformed_launch_grid()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// A finalized protocol action must resolve to a DFB declaration.
// The empty allocation table is synthetic and isolates this precondition.

module attributes {
  ttl.dfb_allocations = [],
  ttl.launch_grid = [1 : i64, 1 : i64]
} {
  func.func @unresolved_dfb_operand(
      %dfb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-error @below {{`ttl-verify-dfb-spsc` requires every DFB protocol action to resolve to `ttl.bind_cb` with `dfb_id` after finalization}}
    %block = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// External-call effects require the same finalized identity as concrete
// protocol operations.

module attributes {
  ttl.dfb_allocations = [],
  ttl.launch_grid = [1 : i64, 1 : i64]
} {
  func.func @unresolved_external_effect(
      %dfb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-error @below {{`ttl-verify-dfb-spsc` requires every DFB protocol action to resolve to `ttl.bind_cb` with `dfb_id` after finalization}}
    ttl.opaque_call "produce" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>] () {header = "effects.hpp"} : () -> ()
    func.return
  }
}

// -----

// One logical DFB must have one finalized physical index.
// The empty allocation table is synthetic and isolates this precondition.

module attributes {
  ttl.dfb_allocations = [],
  ttl.launch_grid = [1 : i64, 1 : i64]
} {
  func.func @inconsistent_physical_indices()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 10 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{logical DFB 10 has inconsistent finalized cb_index values 0 and 1}}
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 10 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }
}
