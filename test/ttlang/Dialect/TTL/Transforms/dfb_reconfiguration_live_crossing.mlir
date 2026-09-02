// Verifies that an incomplete DFB lifecycle retains its allocation through
// every reconfiguration boundary that it crosses.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s --check-prefix=IR
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEBUG

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary0 = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>
#boundary1 = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer]>

// IR: ttl.dfb_reconfiguration_plan = {
// IR-SAME: entry_reconfiguration = 0 : i64
// IR-SAME: entry_reconfiguration = 1 : i64
// IR: %[[LIVE:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// IR: %[[BEFORE:.*]] = ttl.bind_cb{cb_index = 1, block_count = 1} {dfb_id = 1 : index}
// IR: ttl.dfb_reconfiguration
// IR: %[[MIDDLE:.*]] = ttl.bind_cb{cb_index = 1, block_count = 3} {dfb_id = 2 : index}
// IR: ttl.dfb_reconfiguration
// IR: %[[AFTER:.*]] = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 3 : index}

// DEBUG: DFB logical_id=0 bounded=1
// DEBUG: epochs=[{accesses=[0, 1, 2, 3]
// DEBUG-SAME: entry_reconfiguration=initial
// DEBUG-SAME: active_configurations=[initial, 0, 1]
// DEBUG-SAME: terminal_reconfiguration=none
// DEBUG: DFB conflict lhs=0 rhs=3 reason=concurrent-lifetime
// DEBUG: Total DFB count: 2
// DEBUG: DFB assignment: logical DFB 0 -> physical index 0 storage index 1 (bounded)
// DEBUG: DFB assignment: logical DFB 1 -> physical index 1 storage index 0 (bounded)
// DEBUG: DFB assignment: logical DFB 2 -> physical index 1 storage index 0 (bounded)
// DEBUG: DFB assignment: logical DFB 3 -> physical index 1 storage index 0 (bounded)

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %live = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %before = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.opaque_call "live_produce" dfb_dependencies(
        %live : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>]
        () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "before" dfb_dependencies(
        %before : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>,
                     #ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "effects.hpp"} : () -> ()
    ttl.dfb_reconfiguration #boundary0

    %middle = ttl.bind_cb {cb_index = 2, block_count = 3} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.opaque_call "middle" dfb_dependencies(
        %middle : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>,
                     #ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "effects.hpp"} : () -> ()
    ttl.dfb_reconfiguration #boundary1

    %after = ttl.bind_cb {cb_index = 3, block_count = 2} {dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "after" dfb_dependencies(
        %after : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>,
                     #ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "live_consume" dfb_dependencies(
        %live : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "effects.hpp"} : () -> ()
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    ttl.dfb_reconfiguration #boundary0
    ttl.dfb_reconfiguration #boundary1
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    ttl.dfb_reconfiguration #boundary0
    ttl.dfb_reconfiguration #boundary1
    return
  }
}
