// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s

// Verify logical-device coordinate argument indexing and row-major flattening.

// CHECK-LABEL: func.func @current_device_index
// CHECK-DAG: %[[THREE:.*]] = arith.constant 3 : i32
// CHECK-DAG: %[[ROW_ARG_INDEX:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[COL_ARG_INDEX:.*]] = arith.constant 1 : index
// CHECK: %[[ROW:.*]] = ttkernel.get_common_arg_val(%[[ROW_ARG_INDEX]]) : (index) -> i32
// CHECK: %[[COL:.*]] = ttkernel.get_common_arg_val(%[[COL_ARG_INDEX]]) : (index) -> i32
// CHECK: %[[COL_BASE:.*]] = arith.muli %[[ROW]], %[[THREE]] : i32
// CHECK: %[[LINEAR_INDEX:.*]] = arith.addi %[[COL_BASE]], %[[COL]] : i32
// CHECK: %[[RESULT:.*]] = arith.index_cast %[[LINEAR_INDEX]] : i32 to index
// CHECK: return %[[RESULT]] : index
func.func @current_device_index() -> index attributes {
  "ttl.kernel_thread" = #ttkernel.thread<noc>
} {
  %index = ttl.current_device_index
    <components = <name = "device", extent = [2, 3]>> : index
  return %index : index
}

// -----

// Per-kernel computed-address bases precede module-wide pipe resources and
// logical-device coordinates in the common runtime argument sequence.

module attributes {
  ttl.pipe_global_semaphore_count = 2 : i64,
  ttl.pipe_sram_scratch_bytes = 64 : i64
} {
  // CHECK-LABEL: func.func @current_device_index_with_pipe_resources
  // CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : i32
  // CHECK-DAG: %[[ROW_ARG_INDEX:.*]] = arith.constant 6 : index
  // CHECK-DAG: %[[COL_ARG_INDEX:.*]] = arith.constant 7 : index
  // CHECK: %[[ROW:.*]] = ttkernel.get_common_arg_val(%[[ROW_ARG_INDEX]]) : (index) -> i32
  // CHECK: %[[COL:.*]] = ttkernel.get_common_arg_val(%[[COL_ARG_INDEX]]) : (index) -> i32
  // CHECK: %[[COL_BASE:.*]] = arith.muli %[[ROW]], %[[TWO]] : i32
  // CHECK: %[[LINEAR_INDEX:.*]] = arith.addi %[[COL_BASE]], %[[COL]] : i32
  // CHECK: %[[RESULT:.*]] = arith.index_cast %[[LINEAR_INDEX]] : i32 to index
  // CHECK: return %[[RESULT]] : index
  func.func @current_device_index_with_pipe_resources(
      %tensor: tensor<1xi32>) -> index attributes {
    "ttl.kernel_thread" = #ttkernel.thread<noc>,
    ttl.pipe_computed_address_dfb_indices = array<i32: 1, 2>
  } {
    %index = ttl.current_device_index
        <components = <name = "device", extent = [2, 2]>> : index
    return %index : index
  }
}
