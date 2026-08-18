// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verify external fabric manager effects produce one target-binding
// interval per operation-local claim.

// Acquire ownership starts at call entry and release ownership ends when its
// call returns. Uses within that interval do not create additional intervals.
// CHECK-LABEL: func.func @explicit_interval
// CHECK-SAME: ttl.fabric_manager_intervals = [#ttl.fabric_manager_interval<identity = "external.explicit", kind = external, claim = "explicit", routeIndices = [], interferingIntervals = ["external.scoped", "external.conditional"]>]
// CHECK-NEXT: ttkernel.opaque_call "acquire"
// CHECK-NEXT: ttkernel.opaque_call "use"
// CHECK-NEXT: ttkernel.opaque_call "release"
// CHECK-NEXT: return

// A scoped claim owns its manager for one opaque call.
// CHECK-LABEL: func.func @scoped_interval
// CHECK-SAME: ttl.fabric_manager_intervals = [#ttl.fabric_manager_interval<identity = "external.scoped", kind = external, claim = "scoped", routeIndices = [], interferingIntervals = ["external.explicit", "external.conditional"]>]
// CHECK-NEXT: ttkernel.opaque_call "run"
// CHECK-NEXT: return

// A nested scoped claim records the exact launch nodes that execute it.
// CHECK-LABEL: func.func @conditional_scoped_interval
// CHECK-SAME: ttl.fabric_manager_intervals = [#ttl.fabric_manager_interval<identity = "external.conditional", kind = external, claim = "conditional", routeIndices = [], interferingIntervals = ["external.explicit", "external.scoped"], launchNodes = [1, 0]>]
// CHECK: scf.if
// CHECK: ttkernel.opaque_call "run_conditional"

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @explicit_interval() attributes {
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>} {
    ttl.opaque_call "acquire" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "explicit", kind = acquire>],
        header = "fabric.hpp"} : () -> ()
    ttl.opaque_call "use" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "explicit", kind = use>],
        header = "fabric.hpp"} : () -> ()
    ttl.opaque_call "release" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "explicit", kind = release>],
        header = "fabric.hpp"} : () -> ()
    return
  }

  func.func @scoped_interval() attributes {
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>} {
    ttl.opaque_call "run" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "scoped", kind = scoped>],
        header = "fabric.hpp"} : () -> ()
    return
  }

  func.func @conditional_scoped_interval() attributes {
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>} {
    %node_x = ttl.core_x : index
    %selected_x = arith.constant 1 : index
    %selected = arith.cmpi eq, %node_x, %selected_x : index
    scf.if %selected {
      ttl.opaque_call "run_conditional" () {
          fabric_manager_effects = [#ttl.fabric_manager_effect<
              claim = "conditional", kind = scoped>],
          header = "fabric.hpp"} : () -> ()
    }
    return
  }
}
