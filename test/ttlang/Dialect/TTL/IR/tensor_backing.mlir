// Tests supported block-float tensor-backed DFB element types.
// RUN: ttlang-opt --split-input-file %s | FileCheck %s

// BFP4_B tensor backing uses its physical 576-byte tile page size.
module {
  // CHECK-LABEL: func.func @bfp4_tensor_backing
  func.func @bfp4_tensor_backing() {
    // CHECK: ttl.bind_cb{{.*}}byte_size = 576{{.*}}!ttcore.tile<32x32, bfp_bf4>
    // CHECK-NEXT: return
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 576>} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bfp_bf4>, 1>
    return
  }
}

// -----

// BFP8_B tensor backing uses its physical 1088-byte tile page size.
module {
  // CHECK-LABEL: func.func @bfp8_tensor_backing
  func.func @bfp8_tensor_backing() {
    // CHECK: ttl.bind_cb{{.*}}byte_size = 1088{{.*}}!ttcore.tile<32x32, bfp_bf8>
    // CHECK-NEXT: return
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 1088>} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bfp_bf8>, 1>
    return
  }
}
