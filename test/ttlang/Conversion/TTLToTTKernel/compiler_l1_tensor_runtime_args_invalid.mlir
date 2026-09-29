// Tensor-backed SRAM runtime arguments must fit their emitted 32-bit indices.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=false})'

// A tensor backing outside the runtime index range cannot be serialized.
module attributes {ttl.launch_grid = [1, 1], ttl.memory_model = "compiler-sram", ttl.target_arch = #ttcore.arch<blackhole>} {
  // expected-error @below {{'func.func' op tensor backing index exceeds 32-bit runtime metadata}}
  func.func @oversized_backing() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32, ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %storage = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 2147483648, byte_offset = 0, byte_size = 2048>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}

// -----

// Existing common tensor indices must also be representable before rewriting.
module attributes {ttl.launch_grid = [1, 1], ttl.memory_model = "compiler-sram", ttl.target_arch = #ttcore.arch<blackhole>} {
  // expected-error @below {{'func.func' op ttl.crta_indices must contain non-negative 32-bit integers}}
  func.func @oversized_common_index() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32, ttl.base_cta_index = 1 : i32, ttl.crta_indices = [18446744073709551616 : i128]} {
    %storage = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
