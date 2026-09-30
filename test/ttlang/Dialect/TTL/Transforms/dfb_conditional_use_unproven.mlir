// Conditional non-protocol uses whose DFB lifetime stays unproven.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s

// A conditional final-iteration copy after the pop runs outside the DFB
// transaction interval, so the lifetime is not proven.

// CHECK: DFB logical_id=0 bounded=0
// CHECK: node (0,0) lifecycle_completion=incomplete-use-order
// CHECK-SAME: evidence=ttl.copy kernel=@final_iteration_use

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @producer() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>, ttl.noc_index = 0 : i32
  } {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      %slot = ttl.cb_reserve %dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    return
  }

  func.func @final_iteration_use(%input: tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    %final = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      %value = ttl.cb_wait %dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      %is_final = arith.cmpi eq, %iteration, %final : index
      scf.if %is_final {
        %copy = ttl.copy %input, %dfb : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
        ttl.wait %copy : !ttl.transfer_handle<read>
      }
    }
    return
  }
}

// -----

// Under repeated reconfiguration, a conditional copy after the push runs after
// the transaction that the next boundary follows, so the lifetime is not
// proven.

// CHECK: DFB logical_id=0 bounded=0
// CHECK: node (0,0) lifecycle_completion=incomplete-use-order
// CHECK-SAME: evidence=ttl.cb_push kernel=@producer

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#entry = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>
#exit = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer]>


module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @consumer() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #compute
  } {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    %final = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #entry
      %value = ttl.cb_wait %dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %is_final = arith.cmpi eq, %iteration, %final : index
      scf.if %is_final {
        ttl.opaque_call "inspect" dfb_dependencies(
            %dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            dfb_accesses [#ttl.dfb_non_transactional_access<inspect, 0>]
            () {header = "effects.hpp"} : () -> ()
      }
      ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      ttl.dfb_reconfiguration #exit
    }
    return
  }

  func.func @producer(%input: tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>) attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>, ttl.noc_index = 0 : i32,
    ttl.logical_kernel = #reader
  } {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    %final = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #entry
      %slot = ttl.cb_reserve %dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      %is_final = arith.cmpi eq, %iteration, %final : index
      scf.if %is_final {
        %copy = ttl.copy %input, %dfb : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
      }
      ttl.dfb_reconfiguration #exit
    }
    return
  }

  func.func @writer() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>, ttl.noc_index = 1 : i32,
    ttl.logical_kernel = #writer
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

// -----

// Under repeated reconfiguration, a conditional copy inside the transaction
// whose completion is never awaited does not complete before the push.

// CHECK: DFB logical_id=0 bounded=0
// CHECK: node (0,0) lifecycle_completion=incomplete-use-order
// CHECK-SAME: evidence=ttl.copy kernel=@producer

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#entry = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>
#exit = #ttl.dfb_reconfiguration<1, participants[#compute, #reader, #writer]>


module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @consumer() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #compute
  } {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    %final = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #entry
      %value = ttl.cb_wait %dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %is_final = arith.cmpi eq, %iteration, %final : index
      scf.if %is_final {
        ttl.opaque_call "inspect" dfb_dependencies(
            %dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            dfb_accesses [#ttl.dfb_non_transactional_access<inspect, 0>]
            () {header = "effects.hpp"} : () -> ()
      }
      ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      ttl.dfb_reconfiguration #exit
    }
    return
  }

  func.func @producer(%input: tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>) attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>, ttl.noc_index = 0 : i32,
    ttl.logical_kernel = #reader
  } {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    %final = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.dfb_reconfiguration #entry
      %slot = ttl.cb_reserve %dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %is_final = arith.cmpi eq, %iteration, %final : index
      scf.if %is_final {
        %copy = ttl.copy %input, %dfb : (tensor<1x1x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
      }
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      ttl.dfb_reconfiguration #exit
    }
    return
  }

  func.func @writer() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>, ttl.noc_index = 1 : i32,
    ttl.logical_kernel = #writer
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
