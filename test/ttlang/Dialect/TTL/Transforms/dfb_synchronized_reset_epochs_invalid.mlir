// Tests invalid synchronized dataflow-buffer reset contracts.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})'

// Every declared logical-kernel participant must contain the reset.

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @missing_participant()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{'ttl.opaque_call' op synchronized DFB reset is missing a declared logical-kernel participant}}
    ttl.opaque_call "reset" dfb_reset <0, all_local = false, participants[<kind = compute>, <kind = data_movement>]> (%dfb) {header = "reset.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
    return
  }
}

// -----

// All participants in one reset instance must name the same target set.

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @target_mismatch_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute>} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "reset" dfb_reset <0, all_local = false, participants[<kind = compute>, <kind = data_movement>]> (%first) {header = "reset.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
    return
  }

  func.func @target_mismatch_data_movement()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{'ttl.opaque_call' op synchronized DFB reset participants must declare identical target sets}}
    ttl.opaque_call "reset" dfb_reset <0, all_local = false, participants[<kind = compute>, <kind = data_movement>]> (%second) {header = "reset.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
    return
  }
}

// -----

// A reset may execute at most once for one dispatch and launch node.

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @repeated_reset()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      // expected-error @below {{'ttl.opaque_call' op synchronized DFB reset declaration must execute at most once per dispatch and launch node}}
      ttl.opaque_call "reset" dfb_reset <0, all_local = false, participants[<kind = data_movement>]> (%dfb) {header = "reset.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
    }
    return
  }
}

// -----

// Tensor-backed interfaces cannot be reset by this worker-local contract.

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @tensor_backed_reset()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // expected-error @below {{'ttl.opaque_call' op synchronized DFB reset does not support tensor-backed targets}}
    ttl.opaque_call "reset" dfb_reset <0, all_local = false, participants[<kind = data_movement>]> (%dfb) {header = "reset.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> ()
    return
  }
}

// -----

// Independently conditional participants require the same typed condition.

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @condition_mismatch_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : i64
    %value = ttl.opaque_call "compute_active" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
    %active = arith.cmpi ne, %value, %zero : i64
    scf.if %active {
      ttl.opaque_call "reset" dfb_reset <0, all_local = false, participants[<kind = compute>, <kind = data_movement>]> (%dfb) {header = "reset.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
    }
    return
  }

  func.func @condition_mismatch_data_movement()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : i64
    %value = ttl.opaque_call "data_movement_active" () {condition_result = #ttl.dispatch_condition<1, i64>, header = "condition.hpp"} : () -> i64
    %active = arith.cmpi ne, %value, %zero : i64
    scf.if %active {
      // expected-error @below {{'ttl.opaque_call' op synchronized DFB reset participants execute under different structured conditions}}
      ttl.opaque_call "reset" dfb_reset <0, all_local = false, participants[<kind = compute>, <kind = data_movement>]> (%dfb) {header = "reset.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
    }
    return
  }
}

// -----

// Reset calls cannot mix state reset with ordinary protocol effects.

module {
  func.func @effectful_reset()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{'ttl.opaque_call' op synchronized DFB reset cannot declare protocol effects or unknown DFB access}}
    ttl.opaque_call "reset" dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>] dfb_reset <0, all_local = false, participants[<kind = data_movement>]> (%dfb) {header = "reset.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
    return
  }
}

// -----

// A targeted reset requires a nonempty dependency set.

module {
  func.func @targeted_reset_without_targets()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32} {
    // expected-error @below {{'ttl.opaque_call' op targeted synchronized DFB reset requires DFB dependencies}}
    ttl.opaque_call "reset" dfb_reset <0, all_local = false, participants[<kind = data_movement>]> () {header = "reset.hpp"} : () -> ()
    return
  }
}

// -----

// An all-local reset cannot also declare DFB dependencies.

module {
  func.func @all_local_reset_with_targets()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{'ttl.opaque_call' op all-local synchronized DFB reset cannot declare DFB dependencies}}
    ttl.opaque_call "reset" dfb_reset <0, all_local = true, participants[<kind = data_movement>]> (%dfb) {header = "reset.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
    return
  }
}
