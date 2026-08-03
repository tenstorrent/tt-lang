// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

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
