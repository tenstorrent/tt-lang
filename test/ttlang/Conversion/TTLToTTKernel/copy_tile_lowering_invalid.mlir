// RUN: ttlang-opt %s --convert-ttl-to-ttkernel -split-input-file -verify-diagnostics

// Ensure copy_tile conversion fails when no CB attachment is available.
func.func @copy_tile_missing_cb(%t: !ttcore.tile<32x32, f32>, %src_idx: index, %dst_idx: index) {
  // expected-error @+1 {{failed to legalize operation 'ttl.copy_tile' that was explicitly marked illegal}}
  %dst, %dst_tile = ttl.copy_tile %t, %src_idx, %dst_idx
      : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
  func.return
}
