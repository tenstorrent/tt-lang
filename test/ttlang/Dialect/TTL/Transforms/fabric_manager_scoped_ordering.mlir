// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verify scoped external manager calls are sequential within one
// function and remain concurrent across functions.

// CHECK-LABEL: func.func @sibling_conditionals
// CHECK-SAME: ttl.fabric_manager_intervals = [#ttl.fabric_manager_interval<identity = "external.first", kind = external, claim = "first", routeIndices = [], interferingIntervals = [], launchNodes = [0, 0]>, #ttl.fabric_manager_interval<identity = "external.second", kind = external, claim = "second", routeIndices = [], interferingIntervals = [], launchNodes = [0, 0]>]
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @sibling_conditionals() attributes {
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>} {
    %node_x = ttl.core_x : index
    %selected_x = arith.constant 0 : index
    %selected = arith.cmpi eq, %node_x, %selected_x : index
    scf.if %selected {
      ttl.opaque_call "run_first" () {
          fabric_manager_effects = [#ttl.fabric_manager_effect<
              claim = "first", kind = scoped>],
          header = "fabric.hpp"} : () -> ()
    }
    scf.if %selected {
      ttl.opaque_call "run_second" () {
          fabric_manager_effects = [#ttl.fabric_manager_effect<
              claim = "second", kind = scoped>],
          header = "fabric.hpp"} : () -> ()
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @looped_sequence
// CHECK-SAME: ttl.fabric_manager_intervals = [#ttl.fabric_manager_interval<identity = "external.loop.first", kind = external, claim = "loop.first", routeIndices = [], interferingIntervals = [], launchNodes = [0, 0]>, #ttl.fabric_manager_interval<identity = "external.loop.second", kind = external, claim = "loop.second", routeIndices = [], interferingIntervals = [], launchNodes = [0, 0]>]
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @looped_sequence() attributes {
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>} {
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.opaque_call "run_loop_first" () {
          fabric_manager_effects = [#ttl.fabric_manager_effect<
              claim = "loop.first", kind = scoped>],
          header = "fabric.hpp"} : () -> ()
      ttl.opaque_call "run_loop_second" () {
          fabric_manager_effects = [#ttl.fabric_manager_effect<
              claim = "loop.second", kind = scoped>],
          header = "fabric.hpp"} : () -> ()
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @distinct_launch_nodes
// CHECK-SAME: ttl.fabric_manager_intervals = [#ttl.fabric_manager_interval<identity = "external.left", kind = external, claim = "left", routeIndices = [], interferingIntervals = ["external.right"], launchNodes = [0, 0]>, #ttl.fabric_manager_interval<identity = "external.right", kind = external, claim = "right", routeIndices = [], interferingIntervals = ["external.left"], launchNodes = [1, 0]>]
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @distinct_launch_nodes() attributes {
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>} {
    %node_x = ttl.core_x : index
    %left_x = arith.constant 0 : index
    %right_x = arith.constant 1 : index
    %is_left = arith.cmpi eq, %node_x, %left_x : index
    %is_right = arith.cmpi eq, %node_x, %right_x : index
    scf.if %is_left {
      ttl.opaque_call "run_left" () {
          fabric_manager_effects = [#ttl.fabric_manager_effect<
              claim = "left", kind = scoped>],
          header = "fabric.hpp"} : () -> ()
    }
    scf.if %is_right {
      ttl.opaque_call "run_right" () {
          fabric_manager_effects = [#ttl.fabric_manager_effect<
              claim = "right", kind = scoped>],
          header = "fabric.hpp"} : () -> ()
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @multiple_launch_nodes
// CHECK-SAME: ttl.fabric_manager_intervals = [#ttl.fabric_manager_interval<identity = "external.multi.first", kind = external, claim = "multi.first", routeIndices = [], interferingIntervals = ["external.multi.second"], launchNodes = [0, 0, 1, 0]>, #ttl.fabric_manager_interval<identity = "external.multi.second", kind = external, claim = "multi.second", routeIndices = [], interferingIntervals = ["external.multi.first"], launchNodes = [0, 0, 1, 0]>]
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @multiple_launch_nodes() attributes {
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>} {
    %node_x = ttl.core_x : index
    %upper_x = arith.constant 2 : index
    %selected = arith.cmpi ult, %node_x, %upper_x : index
    scf.if %selected {
      ttl.opaque_call "run_multi_first" () {
          fabric_manager_effects = [#ttl.fabric_manager_effect<
              claim = "multi.first", kind = scoped>],
          header = "fabric.hpp"} : () -> ()
    }
    scf.if %selected {
      ttl.opaque_call "run_multi_second" () {
          fabric_manager_effects = [#ttl.fabric_manager_effect<
              claim = "multi.second", kind = scoped>],
          header = "fabric.hpp"} : () -> ()
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @first_function
// CHECK-SAME: ttl.fabric_manager_intervals = [#ttl.fabric_manager_interval<identity = "external.concurrent.first", kind = external, claim = "concurrent.first", routeIndices = [], interferingIntervals = ["external.concurrent.second"]>]
// CHECK-LABEL: func.func @second_function
// CHECK-SAME: ttl.fabric_manager_intervals = [#ttl.fabric_manager_interval<identity = "external.concurrent.second", kind = external, claim = "concurrent.second", routeIndices = [], interferingIntervals = ["external.concurrent.first"]>]
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @first_function() attributes {
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>} {
    ttl.opaque_call "run_concurrent_first" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "concurrent.first", kind = scoped>],
        header = "fabric.hpp"} : () -> ()
    return
  }

  func.func @second_function() attributes {
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>} {
    ttl.opaque_call "run_concurrent_second" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "concurrent.second", kind = scoped>],
        header = "fabric.hpp"} : () -> ()
    return
  }
}
