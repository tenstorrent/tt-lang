// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

#sharded = #ttl.layout<shape = [4, 64], element_type = bf16,
                       buffer = dram, grid = [1, 1], memory = height_sharded>

func.func @invalid_memory_layout(
    %source: tensor<4x64xbf16, #sharded>, %page: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
  // expected-error @below {{'ttl.copy_tensor_page' op source must use interleaved row-major memory}}
  %xf = ttl.copy_tensor_page %source[%page], %dfb
      : tensor<4x64xbf16, #sharded>,
        !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
      -> !ttl.transfer_handle<read>
  return
}

// -----

#system = #ttl.layout<shape = [4, 64], element_type = bf16,
                      buffer = system_memory, grid = [1, 1], memory = interleaved>

func.func @invalid_source_buffer(
    %source: tensor<4x64xbf16, #system>, %page: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
  // expected-error @below {{'ttl.copy_tensor_page' op source buffer must be DRAM or L1}}
  %xf = ttl.copy_tensor_page %source[%page], %dfb
      : tensor<4x64xbf16, #system>,
        !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
      -> !ttl.transfer_handle<read>
  return
}

// -----

#f16 = #ttl.layout<shape = [4, 64], element_type = f16,
                   buffer = dram, grid = [1, 1], memory = interleaved>

func.func @invalid_source_dtype(
    %source: tensor<4x64xf16, #f16>, %page: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
  // expected-error @below {{'ttl.copy_tensor_page' op source must use row-major bf16 or f32 elements, got 'f16'}}
  %xf = ttl.copy_tensor_page %source[%page], %dfb
      : tensor<4x64xf16, #f16>,
        !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
      -> !ttl.transfer_handle<read>
  return
}

// -----

#rank1 = #ttl.layout<shape = [64], element_type = bf16,
                     buffer = dram, grid = [1, 1], memory = interleaved>

func.func @invalid_rank(
    %source: tensor<64xbf16, #rank1>, %page: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
  // expected-error @below {{'ttl.copy_tensor_page' op source must have static rank 2}}
  %xf = ttl.copy_tensor_page %source[%page], %dfb
      : tensor<64xbf16, #rank1>,
        !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
      -> !ttl.transfer_handle<read>
  return
}

// -----

#dynamic = #ttl.layout<shape = [4, 64], element_type = bf16,
                       buffer = dram, grid = [1, 1], memory = interleaved>

func.func @invalid_dynamic_shape(
    %source: tensor<?x64xbf16, #dynamic>, %page: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
  // expected-error @below {{'ttl.copy_tensor_page' op source must have static rank 2}}
  %xf = ttl.copy_tensor_page %source[%page], %dfb
      : tensor<?x64xbf16, #dynamic>,
        !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
      -> !ttl.transfer_handle<read>
  return
}

// -----

#wrong_shape = #ttl.layout<shape = [5, 64], element_type = bf16,
                           buffer = dram, grid = [1, 1], memory = interleaved>

func.func @invalid_layout_shape(
    %source: tensor<4x64xbf16, #wrong_shape>, %page: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
  // expected-error @below {{'ttl.copy_tensor_page' op source layout shape must match tensor shape}}
  %xf = ttl.copy_tensor_page %source[%page], %dfb
      : tensor<4x64xbf16, #wrong_shape>,
        !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
      -> !ttl.transfer_handle<read>
  return
}

// -----

#wrong_element = #ttl.layout<shape = [4, 64], element_type = f32,
                             buffer = dram, grid = [1, 1], memory = interleaved>

func.func @invalid_layout_element_type(
    %source: tensor<4x64xbf16, #wrong_element>, %page: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
  // expected-error @below {{'ttl.copy_tensor_page' op source layout element type must match tensor element type}}
  %xf = ttl.copy_tensor_page %source[%page], %dfb
      : tensor<4x64xbf16, #wrong_element>,
        !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
      -> !ttl.transfer_handle<read>
  return
}

// -----

#provenance = #ttl.layout<shape = [4, 64], element_type = bf16,
                          buffer = dram, grid = [1, 1], memory = interleaved>

func.func @invalid_source_provenance(
    %source: tensor<4x64xbf16, #provenance>, %page: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %derived = builtin.unrealized_conversion_cast %source
      : tensor<4x64xbf16, #provenance> to tensor<4x64xbf16, #provenance>
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
  // expected-error @below {{'ttl.copy_tensor_page' op source must be an entry-block argument of its kernel function}}
  %xf = ttl.copy_tensor_page %derived[%page], %dfb
      : tensor<4x64xbf16, #provenance>,
        !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
      -> !ttl.transfer_handle<read>
  return
}

// -----

#untiled = #ttl.layout<shape = [4, 64], element_type = bf16,
                       buffer = dram, grid = [1, 1], memory = interleaved>

func.func @invalid_untiled_destination(
    %source: tensor<4x64xbf16, #untiled>, %page: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 64], bf16, 1>
  // expected-error @below {{operand #2 must be dataflow buffer with tiled storage}}
  %xf = ttl.copy_tensor_page %source[%page], %dfb
      : tensor<4x64xbf16, #untiled>, !ttl.cb<[1, 64], bf16, 1>
      -> !ttl.transfer_handle<read>
  return
}

// -----

#destination_dtype = #ttl.layout<shape = [4, 64], element_type = bf16,
                                 buffer = dram, grid = [1, 1], memory = interleaved>

func.func @invalid_destination_dtype(
    %source: tensor<4x64xbf16, #destination_dtype>, %page: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<1x32, f32>, 1>
  // expected-error @below {{'ttl.copy_tensor_page' op destination tile data type must match source element type}}
  %xf = ttl.copy_tensor_page %source[%page], %dfb
      : tensor<4x64xbf16, #destination_dtype>,
        !ttl.cb<[1, 1], !ttcore.tile<1x32, f32>, 1>
      -> !ttl.transfer_handle<read>
  return
}

// -----

#byte_count = #ttl.layout<shape = [4, 64], element_type = bf16,
                          buffer = dram, grid = [1, 1], memory = interleaved>

func.func @invalid_byte_count(
    %source: tensor<4x64xbf16, #byte_count>, %page: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<1x32, bf16>, 1>
  // expected-error @below {{'ttl.copy_tensor_page' op source row byte count (128) must equal destination block byte count (64)}}
  %xf = ttl.copy_tensor_page %source[%page], %dfb
      : tensor<4x64xbf16, #byte_count>,
        !ttl.cb<[1, 1], !ttcore.tile<1x32, bf16>, 1>
      -> !ttl.transfer_handle<read>
  return
}

// -----

#missing_thread = #ttl.layout<shape = [4, 64], element_type = bf16,
                              buffer = dram, grid = [1, 1], memory = interleaved>

func.func @invalid_missing_thread(
    %source: tensor<4x64xbf16, #missing_thread>, %page: index) {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
  // expected-error @below {{'ttl.copy_tensor_page' op must be inside a function with 'ttl.kernel_thread' attribute}}
  %xf = ttl.copy_tensor_page %source[%page], %dfb
      : tensor<4x64xbf16, #missing_thread>,
        !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
      -> !ttl.transfer_handle<read>
  return
}

// -----

#wrong_thread = #ttl.layout<shape = [4, 64], element_type = bf16,
                            buffer = dram, grid = [1, 1], memory = interleaved>

func.func @invalid_wrong_thread(
    %source: tensor<4x64xbf16, #wrong_thread>, %page: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
  // expected-error @below {{'ttl.copy_tensor_page' op is only allowed in data movement (noc) threads}}
  %xf = ttl.copy_tensor_page %source[%page], %dfb
      : tensor<4x64xbf16, #wrong_thread>,
        !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
      -> !ttl.transfer_handle<read>
  return
}

// -----

#invalid_thread_attr = #ttl.layout<shape = [4, 64], element_type = bf16,
                                  buffer = dram, grid = [1, 1], memory = interleaved>

func.func @invalid_thread_attribute(
    %source: tensor<4x64xbf16, #invalid_thread_attr>, %page: index)
    attributes {ttl.kernel_thread = "noc"} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
  // expected-error @below {{'ttl.copy_tensor_page' op is only allowed in data movement (noc) threads}}
  %xf = ttl.copy_tensor_page %source[%page], %dfb
      : tensor<4x64xbf16, #invalid_thread_attr>,
        !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
      -> !ttl.transfer_handle<read>
  return
}

// -----

#handle = #ttl.layout<shape = [4, 64], element_type = bf16,
                      buffer = dram, grid = [1, 1], memory = interleaved>

func.func @invalid_transfer_handle_kind(
    %source: tensor<4x64xbf16, #handle>, %page: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
  // expected-error @below {{result #0 must be read transfer handle}}
  %xf = ttl.copy_tensor_page %source[%page], %dfb
      : tensor<4x64xbf16, #handle>,
        !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 1>
      -> !ttl.transfer_handle<write>
  return
}
