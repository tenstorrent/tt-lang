// RUN: ttlang-opt %s --ttkernel-finalize-tensor-runtime-args --verify-diagnostics --split-input-file

// A local accessor must retain structural tensor identity.
func.func @local_accessor_without_runtime_argument()
    attributes {ttl.crta_indices = [0],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %address = arith.constant 4096 : i32
  // expected-error @below {{'ttkernel.LocalTensorAccessor' op requires a structurally visible common runtime argument}}
  %local = ttkernel.LocalTensorAccessor(%address) : (i32) -> !ttkernel.LocalTensorAccessor
  return
}

// -----

// A local accessor requires one statically identified tensor address.
func.func @local_accessor_with_dynamic_index(%index: index)
    attributes {ttl.crta_indices = [0],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %address = ttkernel.get_common_arg_val(%index) : (index) -> i32
  // expected-error @below {{'ttkernel.LocalTensorAccessor' op requires a constant tensor-address index}}
  %local = ttkernel.LocalTensorAccessor(%address) : (i32) -> !ttkernel.LocalTensorAccessor
  return
}

// -----

// Local tensor slots must fall within the function's tensor prefix.
func.func @local_accessor_index_out_of_range()
    attributes {ttl.crta_indices = [0],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %index = arith.constant 1 : index
  %address = ttkernel.get_common_arg_val(%index) : (index) -> i32
  // expected-error @below {{'ttkernel.LocalTensorAccessor' op tensor-address index 1 is outside [0, 1)}}
  %local = ttkernel.LocalTensorAccessor(%address) : (i32) -> !ttkernel.LocalTensorAccessor
  return
}

// -----

// Tensor metadata contains global tensor indices only.
// expected-error @below {{'func.func' op ttl.crta_indices must contain non-negative integer values}}
func.func @negative_global_tensor_index()
    attributes {ttl.crta_indices = [-1],
                ttl.kernel_thread = #ttkernel.thread<noc>} {
  return
}

// -----

// Common runtime argument positions cannot be negative.
func.func @negative_runtime_argument_index()
    attributes {ttl.crta_indices = [0],
                ttl.kernel_thread = #ttkernel.thread<noc>} {
  %index = arith.constant -1 : index
  // expected-error @below {{'ttkernel.get_common_arg_val' op common runtime argument index must be non-negative}}
  %arg = ttkernel.get_common_arg_val(%index) : (index) -> i32
  return
}

// -----

// Tensor-accessor descriptor positions follow the same non-negative contract.
func.func @negative_tensor_accessor_runtime_index()
    attributes {ttl.crta_indices = [0],
                ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cta = arith.constant 0 : i32
  %crta = arith.constant -1 : i32
  // expected-error @below {{'ttkernel.TensorAccessorArgs' op common runtime argument index must be non-negative}}
  %args = ttkernel.TensorAccessorArgs(%cta, %crta)
  return
}
