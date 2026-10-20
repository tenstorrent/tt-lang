// Verifies DFB reuse across a repeated reconfiguration sequence and restoration
// of the initial descriptor at the loop backedge.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s --check-prefix=IR
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEBUG

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary0 = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>
#boundary1 = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer]>

// IR: ttl.dfb_allocations = [
// IR-SAME: block_count = 2 : i32
// IR-SAME: dfb_index = 0 : i32
// IR-SAME: ttl.dfb_reconfiguration_plan = {
// IR-SAME: boundary_ordinals = array<i64: 0, 1>
// IR-SAME: block_count = 2 : i32
// IR-SAME: block_count = 3 : i32
// IR-SAME: entry_reconfiguration = 0 : i64
// IR-SAME: block_count = 2 : i32
// IR-SAME: entry_reconfiguration = 1 : i64

// DEBUG: DFB logical_id=0 bounded=1
// DEBUG: node (0,0) lifecycle_completion=complete
// DEBUG-SAME: descriptor_installations=[initial, 1]
// DEBUG: DFB logical_id=1 bounded=1
// DEBUG: node (0,0) lifecycle_completion=complete
// DEBUG-SAME: entry_reconfiguration=0
// DEBUG: Total DFB count: 1
// DEBUG: DFB assignment: logical DFB 0 -> physical index 0 storage index 0 (bounded)
// DEBUG: DFB assignment: logical DFB 1 -> physical index 0 storage index 0 (bounded)

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 3>
    %lower = arith.constant 0 : index
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.opaque_call "first" dfb_dependencies(
          %first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                       #ttl.dfb_protocol_effect<push, 0, 1>,
                       #ttl.dfb_protocol_effect<wait, 0, 1>,
                       #ttl.dfb_protocol_effect<pop, 0, 1>]
          () {header = "effects.hpp"} : () -> ()
      ttl.dfb_reconfiguration #boundary0
      ttl.opaque_call "second" dfb_dependencies(
          %second : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 3>)
          dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 2>,
                       #ttl.dfb_protocol_effect<push, 0, 2>,
                       #ttl.dfb_protocol_effect<wait, 0, 2>,
                       #ttl.dfb_protocol_effect<pop, 0, 2>]
          () {header = "effects.hpp"} : () -> ()
      ttl.dfb_reconfiguration #boundary1
    }
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %lower = arith.constant 0 : index
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
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
    %lower = arith.constant 0 : index
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #boundary0
      ttl.dfb_reconfiguration #boundary1
    }
    return
  }
}

// -----

// A repeated external-call lifecycle remains unbounded when the
// reconfiguration after its final access preserves state.

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "external_without_discard">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "external_without_discard">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "external_without_discard">
#middle = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>
#done = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer]>

// DEBUG: DFB logical_id=0 bounded=0
// DEBUG-SAME: access_completion_proven=0
// DEBUG: lifecycle_completion=unsupported-control-flow

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @external_without_discard_compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    scf.for %iteration = %lower to %upper step %step {
      %condition = ttl.opaque_call "condition" () {
          condition_result = #ttl.dispatch_condition<0, i64>,
          header = "condition.hpp"} : () -> i64
      %active = arith.cmpi ne, %condition, %zero : i64
      ttl.dfb_reconfiguration #middle
      scf.if %active {
        ttl.opaque_call "after" dfb_dependencies(
            %dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            () {header = "access.hpp"} : () -> ()
      }
      ttl.dfb_reconfiguration #done
    }
    return
  }

  func.func @external_without_discard_read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %lower = arith.constant 0 : index
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #middle
      ttl.dfb_reconfiguration #done
    }
    return
  }

  func.func @external_without_discard_write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    %lower = arith.constant 0 : index
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #middle
      ttl.dfb_reconfiguration #done
    }
    return
  }
}

// -----

// A repeated external-call lifecycle may follow a non-discarding
// reconfiguration when a later reconfiguration restores canonical state.

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "external_live_crossing">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "external_live_crossing">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "external_live_crossing">
#middle = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>
#done = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer], discard_dfb_state = true>

// IR: ttl.dfb_allocations = [
// IR-SAME: dfb_index = 0 : i32
// DEBUG: DFB logical_id=0 bounded=1
// DEBUG-SAME: conditionally_bounded=0
// DEBUG-SAME: access_completion_proven=1
// DEBUG: Total DFB count: 1

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @external_live_crossing_compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    scf.for %iteration = %lower to %upper step %step {
      %condition = ttl.opaque_call "condition" () {
          condition_result = #ttl.dispatch_condition<0, i64>,
          header = "condition.hpp"} : () -> i64
      %active = arith.cmpi ne, %condition, %zero : i64
      ttl.dfb_reconfiguration #middle
      scf.if %active {
        ttl.opaque_call "after" dfb_dependencies(
            %dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            () {header = "access.hpp"} : () -> ()
      }
      ttl.dfb_reconfiguration #done
    }
    return
  }

  func.func @external_live_crossing_read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %lower = arith.constant 0 : index
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #middle
      ttl.dfb_reconfiguration #done
    }
    return
  }

  func.func @external_live_crossing_write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    %lower = arith.constant 0 : index
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #middle
      ttl.dfb_reconfiguration #done
    }
    return
  }
}
