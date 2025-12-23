// RUN: ttlang-opt %s --ttl-insert-tile-regs-sync --verify-diagnostics --split-input-file

#map = affine_map<(d0, d1) -> (d0, d1)>

// Compute op with no inputs should fail init_sfpu insertion.
func.func @compute_no_inputs() -> tensor<1x1x!ttcore.tile<32x32, f32>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %output = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
  %output_cb = ttl.attach_cb %output, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // expected-error @below {{requires CB attachments on inputs and outputs for init_sfpu; missing input CB}}
  %result = ttl.compute ins() outs(%output_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%out: !ttcore.tile<32x32, f32>):
    ttl.yield %out : !ttcore.tile<32x32, f32>
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
}
