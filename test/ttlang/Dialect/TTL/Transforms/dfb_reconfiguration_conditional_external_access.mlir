// Verifies that state-discarding DFB reconfiguration completes conditional
// external accesses, allowing later DFBs to reuse their physical indices.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s --check-prefix=IR
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEBUG

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#entry = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer], discard_dfb_state = true>
#exit = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer], discard_dfb_state = true>

// IR: ttl.dfb_allocations = [
// IR-SAME: dfb_index = 0 : i32
// IR-NOT: dfb_index = 1 : i32

// DEBUG: DFB logical_id=0 bounded=1
// DEBUG: epochs=[{executions=3,accesses=[0, 1]
// DEBUG-SAME: terminal_reconfiguration=0
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
        ttl.opaque_call "conditional" dfb_dependencies(
            %conditional : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
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

// -----

// Logical negation of a node predicate retains an exact access domain. Each
// reconfiguration proves that the preceding access has completed, so the two
// DFBs may use the same physical index.

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "unknown_domain_external">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "unknown_domain_external">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "unknown_domain_external">
#second = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer], discard_dfb_state = true>
#done = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer], discard_dfb_state = true>

// IR: ttl.dfb_allocations = [
// IR-SAME: dfb_index = 0 : i32
// IR-NOT: dfb_index = 1 : i32

// DEBUG: DFB logical_id=0 bounded=1
// DEBUG-SAME: conditionally_bounded=0
// DEBUG: DFB logical_id=1 bounded=1
// DEBUG-SAME: conditionally_bounded=0
// DEBUG: Total DFB count: 1

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @unknown_domain_compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : index
    %node_x = ttl.core_x : index
    %not_selected = arith.cmpi ne, %node_x, %zero : index
    %selected = emitc.logical_not %not_selected : i1
    scf.if %selected {
      ttl.opaque_call "first" dfb_dependencies(
          %first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          () {header = "access.hpp"} : () -> ()
    }
    ttl.dfb_reconfiguration #second
    scf.if %selected {
      ttl.opaque_call "second" dfb_dependencies(
          %second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          () {header = "access.hpp"} : () -> ()
    }
    ttl.dfb_reconfiguration #done
    return
  }

  func.func @unknown_domain_read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    ttl.dfb_reconfiguration #second
    ttl.dfb_reconfiguration #done
    return
  }

  func.func @unknown_domain_write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    ttl.dfb_reconfiguration #second
    ttl.dfb_reconfiguration #done
    return
  }
}

// -----

// Exact zero execution gives the first DFB an empty access domain. The later
// compatible DFB may use the same index because they share no launch node.

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#entry = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer], discard_dfb_state = true>
#exit = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer], discard_dfb_state = true>

// IR: ttl.dfb_allocations = [
// IR-SAME: dfb_index = 0 : i32
// IR-NOT: dfb_index = 1 : i32

// DEBUG: DFB logical_id=0 bounded=0
// DEBUG-SAME: access_completion_proven=0
// DEBUG-SAME: domain={}
// DEBUG: DFB logical_id=1 bounded=1
// DEBUG: Total DFB count: 1

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @exact_zero_compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %inactive = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %later = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %lower = arith.constant 0 : index
    %upper = arith.constant 3 : index
    %step = arith.constant 1 : index
    %core_x = ttl.core_x : index
    scf.for %iteration = %lower to %upper step %step {
      scf.for %inactive_iteration = %lower to %core_x step %step {
        ttl.opaque_call "inactive" dfb_dependencies(
            %inactive : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>)
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

  func.func @exact_zero_read() attributes {
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

  func.func @exact_zero_write() attributes {
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
#exit = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer]>

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
        ttl.opaque_call "conditional" dfb_dependencies(
            %conditional : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
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
