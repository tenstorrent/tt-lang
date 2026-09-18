// Tests storage reuse after a synchronized reset canonicalizes protocol state
// following named opaque external DFB accesses.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s

// Reader and compute execute runtime-dependent external protocols on `old`.
// Their collective reset establishes canonical protocol state without
// asserting internal transaction counts, then permits `current` to reuse the
// physical DFB.

// CHECK: DFB allocation group #ttl.dfb_allocation_group<0> members=[0, 1] envelope_bytes=6144 handoff=proven
// CHECK: DFB logical_id=0 bounded=1{{.*}}opaque_external_access=1
// CHECK: opaque_protocol_reset=1
// CHECK: epochs=[{accesses=[0, 1],transactions=[],write_owner=unknown,read_owner=unknown,entry_reconfiguration=initial,active_configurations=[initial],terminal_reset=0,opaque_protocol_reset=1,terminal_reconfiguration=none,terminal_state=canonical}]
// CHECK: DFB logical_id=1 bounded=1{{.*}}allocation_group=#ttl.dfb_allocation_group<0>
// CHECK: Total DFB count: 1

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @opaque_reset_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "opaque_reset">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %current = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.opaque_call "runtime_chunk_reader" dfb_dependencies(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) () {header = "sdpa.hpp"} : () -> ()
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "opaque_reset">, <kind = data_movement, identity = "reader", operation = "opaque_reset">, <kind = data_movement, identity = "writer", operation = "opaque_reset">]>(%old, %current : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    %slot = ttl.cb_reserve %current
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @opaque_reset_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "opaque_reset">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %current = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.opaque_call "runtime_chunk_compute" dfb_dependencies(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) () {header = "sdpa.hpp"} : () -> ()
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "opaque_reset">, <kind = data_movement, identity = "reader", operation = "opaque_reset">, <kind = data_movement, identity = "writer", operation = "opaque_reset">]>(%old, %current : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    %slot = ttl.cb_wait %current
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @opaque_reset_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "opaque_reset">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %current = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "opaque_reset">, <kind = data_movement, identity = "reader", operation = "opaque_reset">, <kind = data_movement, identity = "writer", operation = "opaque_reset">]>(%old, %current : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    return
  }
}

// -----

// Equal typed predicates prove that the opaque access, matching reset, and
// following lifecycle execute together on each possible launch node.

// CHECK: DFB allocation group #ttl.dfb_allocation_group<0> members=[0, 1] envelope_bytes=6144 handoff=proven
// CHECK: DFB logical_id=0 bounded=1 compiler_created=0 conditionally_bounded=0 opaque_external_access=1
// CHECK: node (0,0) lifecycle_completion=complete domain_assumption=exact conditional_execution=1 opaque_protocol_reset=1
// CHECK: node (1,0) lifecycle_completion=complete domain_assumption=exact conditional_execution=1 opaque_protocol_reset=1
// CHECK: DFB logical_id=1 bounded=1 compiler_created=0 conditionally_bounded=0
// CHECK: Total DFB count: 1

module attributes {ttl.launch_grid = [2, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @conditional_opaque_reset_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "conditional_opaque_reset">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %current = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %zero = arith.constant 0 : i64
    %value = ttl.opaque_call "reader_active" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
    %active = arith.cmpi ne, %value, %zero : i64
    scf.if %active {
      ttl.opaque_call "runtime_chunk_reader" dfb_dependencies(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) () {header = "sdpa.hpp"} : () -> ()
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "conditional_opaque_reset">, <kind = data_movement, identity = "reader", operation = "conditional_opaque_reset">, <kind = data_movement, identity = "writer", operation = "conditional_opaque_reset">]>(%old, %current : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
      %slot = ttl.cb_reserve %current
          : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    }
    return
  }

  func.func @conditional_opaque_reset_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "conditional_opaque_reset">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %current = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %zero = arith.constant 0 : i64
    %value = ttl.opaque_call "compute_active" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
    %active = arith.cmpi ne, %value, %zero : i64
    scf.if %active {
      ttl.opaque_call "runtime_chunk_compute" dfb_dependencies(%old : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) () {header = "sdpa.hpp"} : () -> ()
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "conditional_opaque_reset">, <kind = data_movement, identity = "reader", operation = "conditional_opaque_reset">, <kind = data_movement, identity = "writer", operation = "conditional_opaque_reset">]>(%old, %current : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
      %slot = ttl.cb_wait %current
          : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %current : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    }
    return
  }

  func.func @conditional_opaque_reset_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "conditional_opaque_reset">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %old = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %current = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %zero = arith.constant 0 : i64
    %value = ttl.opaque_call "writer_active" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "condition.hpp"} : () -> i64
    %active = arith.cmpi ne, %value, %zero : i64
    scf.if %active {
      ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "conditional_opaque_reset">, <kind = data_movement, identity = "reader", operation = "conditional_opaque_reset">, <kind = data_movement, identity = "writer", operation = "conditional_opaque_reset">]>(%old, %current : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
    }
    return
  }
}
