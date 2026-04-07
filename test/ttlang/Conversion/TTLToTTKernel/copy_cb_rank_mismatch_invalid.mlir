// RUN: ttlang-opt %s --convert-ttl-to-ttkernel -split-input-file -verify-diagnostics

#layout3d = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                        buffer = dram, grid = [1, 1], memory = interleaved>

// CB shape rank (2) does not match 3D tensor rank.
// expected-error @+1 {{failed to legalize operation 'ttl.copy' that was explicitly marked illegal}}
module {
  func.func @cb_shape_rank_mismatch(
      %arg0: tensor<2x2x2x!ttcore.tile<32x32, f32>, #layout3d>)
      attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0],
                  ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], f32, 1>
    %slice = ttl.tensor_slice %arg0[%c0, %c0, %c0]
        : tensor<2x2x2x!ttcore.tile<32x32, f32>, #layout3d>
          -> tensor<1x1x1x!ttcore.tile<32x32, f32>, #layout3d>
    %xf = ttl.copy %slice, %cb
        : (tensor<1x1x1x!ttcore.tile<32x32, f32>, #layout3d>,
           !ttl.cb<[1, 1], f32, 1>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}
