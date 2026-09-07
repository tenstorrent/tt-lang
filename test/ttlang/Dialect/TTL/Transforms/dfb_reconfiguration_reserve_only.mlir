// Verifies that a reserve-only external access receives the descriptor
// selected for its DFB configuration epoch.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s --check-prefix=IR
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEBUG

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#entry = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer], discard_dfb_state = true>
#exit = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer], discard_dfb_state = true>

// Core (0,0) publishes data, core (1,0) only waits for free DFB capacity, and
// core (2,0) does not access the DFB. The entry configuration must include the
// two accessing cores so both receive the current descriptor.
// IR: ttl.dfb_reconfiguration_plan = {
// IR-SAME: entry_reconfiguration = 0 : i64
// IR-SAME: nodes = {{\[\[0, 0\], \[1, 0\]\]}}
// DEBUG: node (1,0) lifecycle_completion=complete
// DEBUG-SAME: entry_reconfiguration=0

module attributes {ttl.launch_grid = [3, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %buffer = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %publisher = arith.cmpi eq, %core_x, %zero : index
    %capacity_checker = arith.cmpi eq, %core_x, %one : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #entry
      scf.if %publisher {
        ttl.opaque_call "publish" dfb_dependencies(
            %buffer : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
            dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                         #ttl.dfb_protocol_effect<push, 0, 1>]
            () {header = "access.hpp"} : () -> ()
      }
      scf.if %capacity_checker {
        ttl.opaque_call "wait_for_capacity" dfb_dependencies(
            %buffer : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
            dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>]
            () {header = "access.hpp"} : () -> ()
      }
      ttl.dfb_reconfiguration #exit
    }
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #entry
      ttl.dfb_reconfiguration #exit
    }
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #entry
      ttl.dfb_reconfiguration #exit
    }
    return
  }
}
