// RUN: ttlang-opt %s --ttl-lower-dprint-to-emitc --split-input-file --verify-diagnostics
// Verifies tile-mode debug-print operand diagnostics.

// Tile mode requires a tensor containing physical tiles.
func.func @tile_mode_scalar_element(%input: tensor<1x1xbf16>) {
  // expected-error @below {{tile mode operand must have a tile element type; use tensor mode for tensors with scalar element types}}
  // expected-error @below {{failed to legalize operation 'ttl.dprint' that was explicitly marked illegal}}
  "ttl.dprint"(%input) {fmt = "input", mode = "tile", thread = "pack"}
      : (tensor<1x1xbf16>) -> ()
  return
}

// -----

// Tile mode requires a ranked tensor operand.
func.func @tile_mode_non_tensor(%input: index) {
  // expected-error @below {{tile mode operand must be a RankedTensorType; use tensor mode for scalar values}}
  // expected-error @below {{failed to legalize operation 'ttl.dprint' that was explicitly marked illegal}}
  "ttl.dprint"(%input) {fmt = "input", mode = "tile", thread = "pack"}
      : (index) -> ()
  return
}
