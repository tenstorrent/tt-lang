// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s

// Verify that logical device coordinates are flattened in row-major order.

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

// Scratch and global semaphore arguments precede logical device coordinates.
// Verify that device predicates read coordinates after both segments.

// CHECK-LABEL: func.func @device_after_pipe_arguments
// CHECK-DAG: %[[ROW_ARG_INDEX:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[COL_ARG_INDEX:.*]] = arith.constant 5 : index
// CHECK: %[[ROW:.*]] = ttkernel.get_common_arg_val(%[[ROW_ARG_INDEX]]) : (index) -> i32
// CHECK-NEXT: %[[IS_ROW:.*]] = arith.cmpi eq, %[[ROW]], {{.*}} : i32
// CHECK-NEXT: %[[COL:.*]] = ttkernel.get_common_arg_val(%[[COL_ARG_INDEX]]) : (index) -> i32
// CHECK-NEXT: %[[IS_COL:.*]] = arith.cmpi eq, %[[COL]], {{.*}} : i32
// CHECK-NEXT: %[[IS_DEVICE:.*]] = arith.andi %[[IS_ROW]], %[[IS_COL]] : i1
// CHECK-NOT: ttkernel.get_common_arg_val
// CHECK-NEXT: return %[[IS_DEVICE]] : i1
module attributes {
  ttl.pipe_global_semaphore_count = 2 : i64,
  ttl.pipe_sram_scratch_bytes = 32 : i64
} {
  func.func @device_after_pipe_arguments(%tensor: tensor<1xi32>) -> i1
      attributes {
        ttl.kernel_thread = #ttkernel.thread<noc>
      } {
    %is_device = ttl.is_device
        <coordinates = [1, 2]>
        in <components = <name = "mesh", extent = [2, 3]>> : i1
    return %is_device : i1
  }
}
