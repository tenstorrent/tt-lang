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

// -----

// SRAM storage has no physical DFB index for descriptor-based printing.
module attributes {ttl.memory_model = "compiler-sram"} {
  func.func @compiler_sram_cb_print() {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{compiler-sram print supports only scalar mode}}
    "ttl.dprint"(%cb) {fmt = "input", mode = "cb"}
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
    return
  }
}

// -----

// Tensor printing also reads storage through a physical DFB descriptor.
module attributes {ttl.memory_model = "compiler-sram"} {
  func.func @compiler_sram_tensor_print(%input: tensor<1x1xbf16>) {
    // expected-error @below {{compiler-sram print supports only scalar mode}}
    "ttl.dprint"(%input) {fmt = "input", mode = "tensor", num_pages = 1}
        : (tensor<1x1xbf16>) -> ()
    return
  }
}

// -----

// Tile printing also reads storage through a physical DFB descriptor.
module attributes {ttl.memory_model = "compiler-sram"} {
  func.func @compiler_sram_tile_print(%input: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    // expected-error @below {{compiler-sram print supports only scalar mode}}
    "ttl.dprint"(%input) {fmt = "input", mode = "tile"}
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>) -> ()
    return
  }
}

// -----

// Destination-register printing changes pack state on Blackhole.
module attributes {ttl.memory_model = "compiler-sram"} {
  func.func @compiler_sram_dst_print() {
    // expected-error @below {{compiler-sram print supports only scalar mode}}
    "ttl.dprint"() {fmt = "dst", mode = "dst"} : () -> ()
    return
  }
}
