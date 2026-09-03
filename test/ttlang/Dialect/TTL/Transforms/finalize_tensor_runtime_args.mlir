// RUN: ttlang-opt %s --ttkernel-finalize-tensor-runtime-args --split-input-file | FileCheck %s

// Tensor address slots are compacted before compiler-managed arguments.
// CHECK-LABEL: func.func @compact_direct_indices
// CHECK-SAME: ttl.crta_indices = [12 : i32]
// CHECK-SAME: ttl.local_tensor_indices = [12 : i32]
// CHECK: %[[TENSOR_INDEX:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[BANK_BASE:.*]] = ttkernel.get_common_arg_val(%[[TENSOR_INDEX]])
// CHECK-NEXT: ttkernel.LocalTensorAccessor(%[[BANK_BASE]])
// CHECK: %[[COMPILER_INDEX:.*]] = arith.constant 2 : index
// CHECK-NEXT: ttkernel.get_common_arg_val(%[[COMPILER_INDEX]])
// CHECK: arith.constant 2 : i32
// CHECK-NEXT: %[[CRTA_BASE:.*]] = arith.constant 0 : i32
// CHECK-NEXT: ttkernel.TensorAccessorArgs({{.*}}, %[[CRTA_BASE]])
func.func @compact_direct_indices()
    attributes {ttl.crta_indices = [4, 8, 12],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %tensor_index = arith.constant 2 : index
  %bank_base = ttkernel.get_common_arg_val(%tensor_index) : (index) -> i32
  %local = ttkernel.LocalTensorAccessor(%bank_base) : (i32) -> !ttkernel.LocalTensorAccessor
  %compiler_index = arith.constant 4 : index
  %compiler_arg = ttkernel.get_common_arg_val(%compiler_index) : (index) -> i32
  %cta_base = arith.constant 0 : i32
  %crta_base = arith.constant 2 : i32
  %args = ttkernel.TensorAccessorArgs(%cta_base, %crta_base)
  return
}

// -----

// Dynamic tables remap tensor and compiler-managed indices together.
// CHECK-LABEL: func.func @compact_constant_table
// CHECK-SAME: ttl.crta_indices = [20 : i32]
// CHECK-NOT: ttl.local_tensor_indices
// CHECK: ttkernel.experimental.constant_table_lookup %{{.*}}, [0, 1, 2] : index
func.func @compact_constant_table(%selector: index)
    attributes {ttl.crta_indices = [20, 21, 22],
                ttl.kernel_thread = #ttkernel.thread<noc>} {
  %index = ttkernel.experimental.constant_table_lookup %selector, [0, 3, 4] : index
  %arg = ttkernel.get_common_arg_val(%index) : (index) -> i32
  return
}

// -----

// A table used for another purpose is preserved; the runtime-argument use gets
// a separate table with compacted indices.
// CHECK-LABEL: func.func @preserve_shared_constant_table
// CHECK-SAME: ttl.crta_indices = [20 : i32]
// CHECK: %[[ORIGINAL:.*]] = ttkernel.experimental.constant_table_lookup %{{.*}}, [0, 3, 4] : index
// CHECK-NEXT: %[[REMAPPED:.*]] = ttkernel.experimental.constant_table_lookup %{{.*}}, [0, 1, 2] : index
// CHECK-NEXT: ttkernel.get_common_arg_val(%[[REMAPPED]])
// CHECK: return %[[ORIGINAL]] : index
func.func @preserve_shared_constant_table(%selector: index) -> index
    attributes {ttl.crta_indices = [20, 21, 22],
                ttl.kernel_thread = #ttkernel.thread<noc>} {
  %index = ttkernel.experimental.constant_table_lookup %selector, [0, 3, 4] : index
  %arg = ttkernel.get_common_arg_val(%index) : (index) -> i32
  return %index : index
}

// -----

// Verbatim common-argument accesses retain the complete tensor prefix.
// CHECK-LABEL: func.func @preserve_hidden_indices
// CHECK-SAME: ttl.crta_indices = [30 : i32, 31 : i32]
// CHECK-NOT: ttl.local_tensor_indices
func.func @preserve_hidden_indices()
    attributes {ttl.crta_indices = [30, 31],
                ttl.kernel_thread = #ttkernel.thread<noc>} {
  emitc.verbatim "auto address = get_common_arg_val<uint32_t>(1);"
  return
}

// -----

// A structurally visible but dynamic index also retains the complete prefix.
// CHECK-LABEL: func.func @preserve_unresolved_index
// CHECK-SAME: ttl.crta_indices = [40 : i32, 41 : i32]
func.func @preserve_unresolved_index(%index: index)
    attributes {ttl.crta_indices = [40, 41],
                ttl.kernel_thread = #ttkernel.thread<noc>} {
  %arg = ttkernel.get_common_arg_val(%index) : (index) -> i32
  return
}
