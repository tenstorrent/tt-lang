// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -convert-ttl-to-ttkernel

// Summary: Verify malformed external fabric manager ownership declarations are
// rejected before target lowering mutates the module.

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @missing_logical_kernel()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{fabric manager effects require an enclosing logical kernel function}}
    ttl.opaque_call "run" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "manager", kind = scoped>],
        header = "fabric.hpp"} : () -> ()
    return
  }
}

// -----

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @nested_effect() attributes {
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>} {
    %condition = arith.constant true
    scf.if %condition {
      // expected-error @below {{fabric manager effects must be in the logical kernel's straight-line entry block unless the effect is scoped}}
      ttl.opaque_call "run" () {
          fabric_manager_effects = [#ttl.fabric_manager_effect<
              claim = "manager", kind = acquire>],
          header = "fabric.hpp"} : () -> ()
    }
    return
  }
}

// -----

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @first_kernel() attributes {
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>} {
    ttl.opaque_call "run" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "manager", kind = scoped>],
        header = "fabric.hpp"} : () -> ()
    return
  }
  func.func @second_kernel() attributes {
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>} {
    // expected-error @below {{fabric manager claim 'manager' is used by multiple logical kernels}}
    ttl.opaque_call "run" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "manager", kind = scoped>],
        header = "fabric.hpp"} : () -> ()
    return
  }
}

// -----

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @multiple_effects() attributes {
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>} {
    // expected-error @below {{fabric manager claim 'manager' has multiple effects on one call}}
    ttl.opaque_call "run" () {
        fabric_manager_effects = [
          #ttl.fabric_manager_effect<claim = "manager", kind = acquire>,
          #ttl.fabric_manager_effect<claim = "manager", kind = release>],
        header = "fabric.hpp"} : () -> ()
    return
  }
}

// -----

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @mixed_scoped_and_explicit() attributes {
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>} {
    // expected-error @below {{fabric manager claim 'manager' must use one scoped effect or one acquire/release interval, not both}}
    ttl.opaque_call "run" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "manager", kind = scoped>],
        header = "fabric.hpp"} : () -> ()
    ttl.opaque_call "acquire" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "manager", kind = acquire>],
        header = "fabric.hpp"} : () -> ()
    ttl.opaque_call "release" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "manager", kind = release>],
        header = "fabric.hpp"} : () -> ()
    return
  }
}

// -----

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @missing_acquire() attributes {
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>} {
    // expected-error @below {{fabric manager claim 'manager' has no acquire effect}}
    ttl.opaque_call "release" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "manager", kind = release>],
        header = "fabric.hpp"} : () -> ()
    return
  }
}

// -----

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @missing_release() attributes {
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>} {
    // expected-error @below {{fabric manager claim 'manager' has no release effect}}
    ttl.opaque_call "acquire" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "manager", kind = acquire>],
        header = "fabric.hpp"} : () -> ()
    return
  }
}

// -----

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @multiple_acquires() attributes {
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>} {
    // expected-error @below {{fabric manager claim 'manager' has multiple acquire effects}}
    ttl.opaque_call "acquire_first" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "manager", kind = acquire>],
        header = "fabric.hpp"} : () -> ()
    ttl.opaque_call "acquire_second" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "manager", kind = acquire>],
        header = "fabric.hpp"} : () -> ()
    ttl.opaque_call "release" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "manager", kind = release>],
        header = "fabric.hpp"} : () -> ()
    return
  }
}

// -----

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @multiple_releases() attributes {
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>} {
    // expected-error @below {{fabric manager claim 'manager' has multiple release effects}}
    ttl.opaque_call "acquire" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "manager", kind = acquire>],
        header = "fabric.hpp"} : () -> ()
    ttl.opaque_call "release_first" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "manager", kind = release>],
        header = "fabric.hpp"} : () -> ()
    ttl.opaque_call "release_second" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "manager", kind = release>],
        header = "fabric.hpp"} : () -> ()
    return
  }
}

// -----

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @release_before_acquire() attributes {
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>} {
    // expected-error @below {{fabric manager claim 'manager' release must follow its acquire}}
    ttl.opaque_call "release" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "manager", kind = release>],
        header = "fabric.hpp"} : () -> ()
    ttl.opaque_call "acquire" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "manager", kind = acquire>],
        header = "fabric.hpp"} : () -> ()
    return
  }
}

// -----

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @use_outside_interval() attributes {
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>} {
    // expected-error @below {{fabric manager claim 'manager' use must be between its acquire and release}}
    ttl.opaque_call "use" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "manager", kind = use>],
        header = "fabric.hpp"} : () -> ()
    ttl.opaque_call "acquire" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "manager", kind = acquire>],
        header = "fabric.hpp"} : () -> ()
    ttl.opaque_call "release" () {
        fabric_manager_effects = [#ttl.fabric_manager_effect<
            claim = "manager", kind = release>],
        header = "fabric.hpp"} : () -> ()
    return
  }
}
