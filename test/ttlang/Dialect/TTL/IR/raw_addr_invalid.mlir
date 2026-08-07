// Verify raw_addr accepts only the enclosing kernel function's tensor arguments.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>

func.func @raw_addr_rejects_slice(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
  %slice = tensor.cast %arg0 : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> to tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
  // expected-error @below {{'ttl.raw_addr' op operand must be a function tensor argument with TTL layout encoding; slices/views are not supported}}
  %addr = ttl.raw_addr %slice : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> i32
  return
}

// -----

// A nested region entry argument does not have a runtime function-argument slot.
#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>
func.func @raw_addr_rejects_region_argument(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %index = %c0 to %c1 step %c1 iter_args(%tensor_arg = %arg0) -> (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) {
    // expected-error @below {{'ttl.raw_addr' op operand must be a function tensor argument with TTL layout encoding; slices/views are not supported}}
    %addr = ttl.raw_addr %tensor_arg : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> i32
    scf.yield %tensor_arg : tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
  }
  return
}
