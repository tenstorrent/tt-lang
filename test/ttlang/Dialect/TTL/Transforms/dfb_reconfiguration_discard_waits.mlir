// Verifies wait protocols accepted before state-discarding DFB reconfiguration.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s --check-prefix=IR
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEBUG

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#reconfigure = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer], discard_dfb_state = true>

// The producer publishes one page. Compute waits without consuming it. The
// reconfiguration call resets that RISC-local wait state before index reuse.
// IR: ttl.dfb_allocations = [
// IR-SAME: dfb_index = 0 : i32
// IR-NOT: dfb_index = 1 : i32
// DEBUG: DFB logical_id=0 bounded=1
// DEBUG: DFB logical_id=1 bounded=1
// DEBUG: Total DFB count: 1

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "wait_without_pop" dfb_dependencies(
        %stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>]
        () {header = "access.hpp"} : () -> ()
    ttl.dfb_reconfiguration #reconfigure
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %later = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "publish" dfb_dependencies(
        %stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>]
        () {header = "access.hpp"} : () -> ()
    ttl.dfb_reconfiguration #reconfigure
    ttl.opaque_call "publish_later" dfb_dependencies(
        %later : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>]
        () {header = "access.hpp"} : () -> ()
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    %later = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.dfb_reconfiguration #reconfigure
    ttl.opaque_call "consume_later" dfb_dependencies(
        %later : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "access.hpp"} : () -> ()
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#reconfigure = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer], discard_dfb_state = true>

// Existing cumulative queue analysis permits producer progress to alternate
// with consumer progress when total publication exceeds physical capacity.
// State-discard analysis must retain that result rather than replace it.
// IR: ttl.dfb_allocations = [
// IR-SAME: dfb_index = 0 : i32
// IR-NOT: dfb_index = 1 : i32
// DEBUG: DFB logical_id=0 bounded=1
// DEBUG: DFB logical_id=1 bounded=1
// DEBUG: Total DFB count: 1

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    ttl.dfb_reconfiguration #reconfigure
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %later = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "cumulative_publish" dfb_dependencies(
        %first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 2>,
                     #ttl.dfb_protocol_effect<push, 0, 1>,
                     #ttl.dfb_protocol_effect<reserve, 0, 2>,
                     #ttl.dfb_protocol_effect<push, 0, 1>,
                     #ttl.dfb_protocol_effect<reserve, 0, 2>,
                     #ttl.dfb_protocol_effect<push, 0, 1>]
        () {header = "access.hpp"} : () -> ()
    ttl.dfb_reconfiguration #reconfigure
    ttl.opaque_call "publish_later" dfb_dependencies(
        %later : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>]
        () {header = "access.hpp"} : () -> ()
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %later = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "cumulative_consume" dfb_dependencies(
        %first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>,
                     #ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>,
                     #ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "access.hpp"} : () -> ()
    ttl.dfb_reconfiguration #reconfigure
    ttl.opaque_call "consume_later" dfb_dependencies(
        %later : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "access.hpp"} : () -> ()
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#reconfigure = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer], discard_dfb_state = true>

// A producer sequenced after a wait in the same kernel cannot make that wait
// complete. The two logical DFBs therefore cannot share an index.
// IR: ttl.dfb_allocations = [
// IR-SAME: dfb_index = 0 : i32
// IR-SAME: dfb_index = 1 : i32
// DEBUG: DFB logical_id=0 bounded=0
// DEBUG: DFB logical_id=1 bounded=1
// DEBUG: Total DFB count: 2

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %later = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "wait_before_publish" dfb_dependencies(
        %stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>]
        () {header = "access.hpp"} : () -> ()
    ttl.opaque_call "publish_after_wait" dfb_dependencies(
        %stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>]
        () {header = "access.hpp"} : () -> ()
    ttl.dfb_reconfiguration #reconfigure
    ttl.opaque_call "later" dfb_dependencies(
        %later : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>,
                     #ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "access.hpp"} : () -> ()
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    ttl.dfb_reconfiguration #reconfigure
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    ttl.dfb_reconfiguration #reconfigure
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#reconfigure = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer], discard_dfb_state = true>

// Reconfiguration cannot make a two-page wait complete after the producer
// publishes only one page. The two logical DFBs therefore cannot share an
// index.
// IR: ttl.dfb_allocations = [
// IR-SAME: dfb_index = 0 : i32
// IR-SAME: dfb_index = 1 : i32
// DEBUG: DFB logical_id=0 bounded=0
// DEBUG: lifecycle_completion=missing-protocol-effect
// DEBUG: DFB logical_id=1 bounded=1
// DEBUG: Total DFB count: 2

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "wait_for_too_many_pages" dfb_dependencies(
        %stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 2>]
        () {header = "access.hpp"} : () -> ()
    ttl.dfb_reconfiguration #reconfigure
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %stale = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %later = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "publish" dfb_dependencies(
        %stale : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>]
        () {header = "access.hpp"} : () -> ()
    ttl.dfb_reconfiguration #reconfigure
    ttl.opaque_call "publish_later" dfb_dependencies(
        %later : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>]
        () {header = "access.hpp"} : () -> ()
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    %later = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.dfb_reconfiguration #reconfigure
    ttl.opaque_call "consume_later" dfb_dependencies(
        %later : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "access.hpp"} : () -> ()
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#reconfigure = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer], discard_dfb_state = true>

// Compute and the writer may both wait, but only the writer advances the read
// pointer. Reconfiguration resets both RISC-local wait counters.
// IR: ttl.dfb_allocations = [
// IR-SAME: dfb_index = 0 : i32
// IR-NOT: dfb_index = 1 : i32
// DEBUG: DFB logical_id=0 bounded=1
// DEBUG: DFB logical_id=1 bounded=1
// DEBUG: Total DFB count: 1

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %shared = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "compute_wait" dfb_dependencies(
        %shared : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>]
        () {header = "access.hpp"} : () -> ()
    ttl.dfb_reconfiguration #reconfigure
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %shared = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %later = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "publish" dfb_dependencies(
        %shared : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>]
        () {header = "access.hpp"} : () -> ()
    ttl.dfb_reconfiguration #reconfigure
    ttl.opaque_call "publish_later" dfb_dependencies(
        %later : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>]
        () {header = "access.hpp"} : () -> ()
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    %shared = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %later = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "writer_wait_and_pop" dfb_dependencies(
        %shared : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "access.hpp"} : () -> ()
    ttl.dfb_reconfiguration #reconfigure
    ttl.opaque_call "consume_later" dfb_dependencies(
        %later : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "access.hpp"} : () -> ()
    return
  }
}
