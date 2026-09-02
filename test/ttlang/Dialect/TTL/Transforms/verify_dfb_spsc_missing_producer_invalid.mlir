// Summary: Rejects DFB waits without any matching push action.

// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices,ttl-verify-dfb-spsc)'
// RUN: env TTL_RELAX_DFB_SPSC=1 ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices,ttl-verify-dfb-spsc)'

// Reserving DFB storage does not publish data to a waiting consumer.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_reserve %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }

  func.func @consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{logical DFB 0 is waited on but no kernel thread pushes it}}
    // expected-note @below {{a DFB wait blocks until a matching push publishes data}}
    %block = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// A push outside a kernel thread does not participate in the device protocol.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @helper() {
    // expected-note @+1 {{dataflow buffer declared here}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_reserve %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }

  func.func @consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{logical DFB 2 is waited on but no kernel thread pushes it}}
    // expected-note @below {{a DFB wait blocks until a matching push publishes data}}
    %block = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// A typed external wait has the same structural producer requirement.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{logical DFB 1 is waited on but no kernel thread pushes it}}
    // expected-note @below {{a DFB wait blocks until a matching push publishes data}}
    ttl.opaque_call "consume" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>] () {header = "effects.hpp"} : () -> ()
    func.return
  }
}

// -----

// An inspection contract excludes producer protocol inside the external call.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @inspection() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "inspect" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_accesses [#ttl.dfb_non_transactional_access<inspect, 0>] () {header = "effects.hpp"} : () -> ()
    func.return
  }

  func.func @consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{logical DFB 3 is waited on but no kernel thread pushes it}}
    // expected-note @below {{a DFB wait blocks until a matching push publishes data}}
    %block = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}
