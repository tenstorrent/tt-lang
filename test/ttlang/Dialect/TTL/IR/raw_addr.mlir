// Verify raw_addr round-trip parsing/printing.
// RUN: ttlang-opt %s | FileCheck %s

#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>

// CHECK-LABEL: func.func @raw_addr_on_tensor_arg
// CHECK: = ttl.raw_addr %arg0 : tensor<1x1x!ttcore.tile<32x32, f32>, #ttl.layout<{{.*}}>> -> i32
func.func @raw_addr_on_tensor_arg(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
  %addr = ttl.raw_addr %arg0 : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> i32
  return
}

// CHECK-LABEL: func.func @raw_addr_on_compute_tensor_arg
// CHECK: = ttl.raw_addr %arg0 : tensor<1x1x!ttcore.tile<32x32, f32>, #ttl.layout<{{.*}}>> -> i32
func.func @raw_addr_on_compute_tensor_arg(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %addr = ttl.raw_addr %arg0 : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> i32
  return
}
