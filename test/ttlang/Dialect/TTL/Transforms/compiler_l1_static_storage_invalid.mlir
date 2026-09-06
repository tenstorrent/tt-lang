// Rejects epoch-dependent ownership before compiler-managed L1 placement.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1})'

// Reconfiguration cannot authorize sharing under the static ownership contract.
#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>



module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.opaque_call "first" dfb_dependencies(
        %first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>,
                     #ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "effects.hpp"} : () -> ()
    // expected-error @below {{compiler-l1 requires static storage ownership; DFB reset and reconfiguration are unsupported}}
    ttl.dfb_reconfiguration #boundary
    %second = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.opaque_call "second" dfb_dependencies(
        %second : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>,
                     #ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "effects.hpp"} : () -> ()
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    ttl.dfb_reconfiguration #boundary
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    ttl.dfb_reconfiguration #boundary
    return
  }
}

// -----

// Selected resets require a control-state transition that this backend cannot emit.
module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @unconditional_reset_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %current = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %old_slot = ttl.cb_reserve %old : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %old : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    // expected-error @below {{compiler-l1 requires static storage ownership; DFB reset and reconfiguration are unsupported}}
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    %current_slot = ttl.cb_reserve %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @unconditional_reset_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %current = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    %current_slot = ttl.cb_wait %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @unconditional_reset_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    return
  }
}


// -----

// Resetting all interfaces has the same unsupported control-state requirement.
module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @unconditional_reset_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %current = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %old_slot = ttl.cb_reserve %old : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %old : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    // expected-error @below {{compiler-l1 requires static storage ownership; DFB reset and reconfiguration are unsupported}}
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>
    %current_slot = ttl.cb_reserve %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @unconditional_reset_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %current = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>
    %current_slot = ttl.cb_wait %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @unconditional_reset_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>
    return
  }
}
