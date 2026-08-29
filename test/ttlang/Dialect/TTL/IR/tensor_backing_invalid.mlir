// Tests verifier diagnostics for tensor-backed DFB ranges.
// RUN: ttlang-opt --verify-diagnostics --split-input-file %s

// The backing range must name a tensor argument.
module {
  func.func @negative_tensor_index() {
    // expected-error @below {{tensor_index must be non-negative}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = -1, byte_offset = 0, byte_size = 2048>} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}

// -----

// A negative offset cannot identify a byte range within a tensor shard.
module {
  func.func @negative_byte_offset() {
    // expected-error @below {{byte_offset must be non-negative}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = -1, byte_size = 2048>} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}

// -----

// A zero-sized range cannot provide dataflow buffer storage.
module {
  func.func @zero_byte_size() {
    // expected-error @below {{byte_size must be positive}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 0>} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}

// -----

// A negative size cannot identify a byte range within a tensor shard.
module {
  func.func @negative_byte_size() {
    // expected-error @below {{byte_size must be positive}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = -1>} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}

// -----

// The backing offset must satisfy the public descriptor interface.
module {
  func.func @unaligned_byte_offset() {
    // expected-error @below {{tensor backing byte_offset must be aligned to the 2048-byte dataflow buffer page size}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 1, byte_size = 2048>} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}

// -----

// The bound range represents the complete DFB capacity.
module {
  func.func @incomplete_capacity() {
    // expected-error @below {{tensor backing byte_size must equal the complete dataflow buffer capacity (expected 4096, got 2048)}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// The range end must fit the uint32 API accepted by TTNN.
module {
  func.func @range_end_overflow() {
    // expected-error @below {{byte_offset and byte_size must fit the uint32 descriptor fields}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 4294967295, byte_size = 2048>} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}

// -----

// A sub-byte scalar type is not a valid tensor-backed page format.
module {
  func.func @i1_element_type() {
    // expected-error @below {{tensor backing requires a TTCore tile element type}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 1>} : !ttl.cb<[1, 1], i1, 1>
    return
  }
}

// -----

// An i4 element type is also smaller than one byte.
module {
  func.func @i4_element_type() {
    // expected-error @below {{tensor backing requires a TTCore tile element type}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 1>} : !ttl.cb<[1, 1], i4, 1>
    return
  }
}

// -----

// An index type has no fixed device page width.
module {
  func.func @index_element_type() {
    // expected-error @below {{tensor backing requires a TTCore tile element type}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 1>} : !ttl.cb<[1, 1], index, 1>
    return
  }
}

// -----

// Tensor-backed formats reject tile data types outside the supported set.
module {
  func.func @unsupported_tile_data_type() {
    // expected-error @below {{tensor backing supports only BF16, FP32, BFP4_B, and BFP8_B tile element types}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f16>, 1>
    return
  }
}
