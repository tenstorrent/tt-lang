// Verifies the access contracts accepted before repeated state-discarding DFB
// reconfiguration.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s --check-prefix=IR
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEBUG

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#entry = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer], discard_dfb_state = true>
#exit = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer], discard_dfb_state = true>

// An untyped conditional external access can end at a state-discarding
// reconfiguration. A typed inspection in the same lifecycle remains explicit
// non-transactional use rather than being mistaken for an unscoped access.
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
    %conditional = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %later = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %lower = arith.constant 0 : index
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    %selected = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      %condition = arith.cmpi eq, %iteration, %selected : index
      scf.if %condition {
        ttl.opaque_call "conditional_unknown" dfb_dependencies(
            %conditional : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            () {header = "access.hpp"} : () -> ()
        ttl.opaque_call "conditional_inspection" dfb_dependencies(
            %conditional : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            dfb_accesses [#ttl.dfb_non_transactional_access<inspect, 0>]
            () {header = "access.hpp"} : () -> ()
      }
      ttl.dfb_reconfiguration #entry
      ttl.opaque_call "later" dfb_dependencies(
          %later : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
          dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                       #ttl.dfb_protocol_effect<push, 0, 1>,
                       #ttl.dfb_protocol_effect<wait, 0, 1>,
                       #ttl.dfb_protocol_effect<pop, 0, 1>]
          () {header = "access.hpp"} : () -> ()
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
    %upper = arith.constant 3 : index
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
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #entry
      ttl.dfb_reconfiguration #exit
    }
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#entry = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer], discard_dfb_state = true>
#exit = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer], discard_dfb_state = true>

// A conditional consumer can block before the boundary, so state discard
// cannot replace an exact execution proof.
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
    %conditional = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %later = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %lower = arith.constant 0 : index
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    %selected = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      %condition = arith.cmpi eq, %iteration, %selected : index
      scf.if %condition {
        ttl.opaque_call "conditional_consumer" dfb_dependencies(
            %conditional : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>,
                         #ttl.dfb_protocol_effect<pop, 0, 1>]
            () {header = "access.hpp"} : () -> ()
      }
      ttl.dfb_reconfiguration #entry
      ttl.opaque_call "later" dfb_dependencies(
          %later : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
          dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                       #ttl.dfb_protocol_effect<push, 0, 1>,
                       #ttl.dfb_protocol_effect<wait, 0, 1>,
                       #ttl.dfb_protocol_effect<pop, 0, 1>]
          () {header = "access.hpp"} : () -> ()
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
    %upper = arith.constant 3 : index
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
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #entry
      ttl.dfb_reconfiguration #exit
    }
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#entry = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer], discard_dfb_state = true>
#exit = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer], discard_dfb_state = true>

// A state-discarding boundary cannot make a producer safe when its static
// maximum can exceed DFB capacity before the call returns.
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
    %conditional = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %later = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %lower = arith.constant 0 : index
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    %selected = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      %condition = arith.cmpi eq, %iteration, %selected : index
      scf.if %condition {
        ttl.opaque_call "conditional_producer" dfb_dependencies(
            %conditional : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
            dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                         #ttl.dfb_protocol_effect<push, 0, 1>,
                         #ttl.dfb_protocol_effect<reserve, 0, 1>,
                         #ttl.dfb_protocol_effect<push, 0, 1>]
            () {header = "access.hpp"} : () -> ()
      }
      ttl.dfb_reconfiguration #entry
      ttl.opaque_call "later" dfb_dependencies(
          %later : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
          dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                       #ttl.dfb_protocol_effect<push, 0, 1>,
                       #ttl.dfb_protocol_effect<wait, 0, 1>,
                       #ttl.dfb_protocol_effect<pop, 0, 1>]
          () {header = "access.hpp"} : () -> ()
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
    %upper = arith.constant 3 : index
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
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #entry
      ttl.dfb_reconfiguration #exit
    }
    return
  }
}
