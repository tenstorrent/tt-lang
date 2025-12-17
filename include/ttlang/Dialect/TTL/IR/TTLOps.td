// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPS_TD
#define TTLANG_DIALECT_TTL_IR_TTLOPS_TD

include "ttlang/Dialect/TTL/IR/TTLBase.td"
include "ttlang/Dialect/TTL/IR/TTLOpsTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TTL operation definitions
//===----------------------------------------------------------------------===//

def TTL_CreateCBOp : TTL_Op<"create_cb", [Pure]> {
  let summary = "Create a circular buffer value";
  let description = [{
    Creates a TTL circular buffer (`!ttl.cb`) representing an L1-resident buffer
    used for block-level communication between TT-Lang threads.

    Circular buffers are created in the kernel scope and used inside data-movement
    and compute threads via higher-level operations (for example `ttl.copy` in the
    DMA MVP).

    - `shape` describes the block shape in "shape units" (tiles for tiled tensors,
      scalars for row-major tensors).
    - `element_type` is the element type stored in the buffer.
    - `buffer_factor` is the number of blocks (for example 2 for double buffering).

    Example:

    ```mlir
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    ```
  }];
  let arguments = (ins
    I64ArrayAttr:$shape,
    TypeAttr:$element_type,
    I64Attr:$buffer_factor,
    OptionalAttr<I32Attr>:$buffer_index,
    OptionalAttr<I64Attr>:$page_size
  );
  let results = (outs TTL_CircularBuffer:$result);
  let assemblyFormat = "`(` `)` attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def TTL_CopyOp : TTL_Op<"copy", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Asynchronous copy between tensor and circular buffer";
  let description = [{
    `ttl.copy` initiates an asynchronous transfer and returns a transfer handle
    (`!ttl.transfer_handle`) that can be synchronized with `ttl.wait`.

    This operation is non-blocking. The destination is not safe to use until a
    corresponding `ttl.wait` has completed.

    In the current MVP (no pipes/blocks yet), exactly one operand must be a
    circular buffer (`!ttl.cb`). The other operand must be a ranked tensor
    carrying a TTNN layout encoding (for example `#ttnn.ttnn_layout<...>`), so
    lowering can derive tile and addressing information.

    TODO(ttl): Add an optional TRID attribute (range 0..15) when TRID-specific
    ttkernel noc ops land in tt-mlir. Issue: #87.

    Example:

    ```mlir
    #dram = #ttnn.buffer_type<dram>
    #layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

    func.func @dma_single(%t: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
      %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
      %xf = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle
      ttl.wait %xf
      func.return
    }
    ```
  }];
  let arguments = (ins
    AnyType:$src,
    AnyType:$dst
  );
  let results = (outs TTL_TransferHandle:$xf);
  let assemblyFormat = "$src `,` $dst attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TTL_WaitOp : TTL_Op<"wait", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Wait on a transfer handle";
  let description = [{
    `ttl.wait` blocks until the asynchronous transfer identified by the input
    transfer handle (`!ttl.transfer_handle`) is complete and the destination is safe to use.

    In the current MVP lowering, this is implemented using a TTKernel global NOC
    barrier as a placeholder until per-transfer (TRID-based) synchronization is
    used.

    TODO(ttl): Add an optional TRID attribute (range 0..15) when TRID-specific
    barriers land in ttkernel (tt-mlir). Issue: #87.

    Example:

    ```mlir
    %xf = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ...
    ttl.wait %xf : !ttl.transfer_handle<read>
    ```

    Note: Transfer direction is modeled in the transfer handle type
    (`!ttl.transfer_handle<read>` or `!ttl.transfer_handle<write>`). The
    verifier requires direction-typed handles so lowering can always emit
    exactly one barrier (read vs. write).
  }];
  // NOTE: Use AnyType here so assembly prints/parses the full type spelling
  // (`!ttl.transfer_handle<read/write>`) rather than the stripped custom form
  // (`<read>/<write>`) and does not require a custom printer. The verifier still
  // enforces the operand is a direction-typed transfer handle.
  let arguments = (ins AnyType:$xf);
  let assemblyFormat = "$xf attr-dict `:` type($xf)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Elementwise compute operations (shared ElementWise base)
//===----------------------------------------------------------------------===//

class TTL_ElementWiseBase<string mnemonic, list<Trait> traits>
    : TTL_Op<mnemonic, traits> {
  let results = (outs AnyType:$result);
}

class TTL_ElementWiseBinary<string mnemonic, list<Trait> traits = []>
    : TTL_ElementWiseBase<
          mnemonic,
          !listconcat(traits, [Pure])> {
  let summary = "Binary elementwise operation";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

class TTL_ElementWiseUnary<string mnemonic, list<Trait> traits = []>
    : TTL_ElementWiseBase<
          mnemonic,
          !listconcat(traits, [Pure])> {
  let summary = "Unary elementwise operation";
  let arguments = (ins AnyType:$input);
  let assemblyFormat =
      "$input attr-dict `:` type($input) `->` type($result)";
}

// Tensor and tile elementwise ops are defined together using multiclass below

//===----------------------------------------------------------------------===//
// Structured Compute Operation (ttl.compute)
//===----------------------------------------------------------------------===//

def TTL_ComputeOp : TTL_Op<"compute", [
    AttrSizedOperandSegments,
    SingleBlockImplicitTerminator<"YieldOp">,
    DeclareOpInterfaceMethods<TilingInterface,
      ["getLoopIteratorTypes", "getIterationDomain", "getTiledImplementation",
       "getResultTilePosition"]>,
    DeclareOpInterfaceMethods<DestinationStyleOpInterface>
  ]> {
  let summary = "Structured tile-level compute operation";
  let description = [{
    `ttl.compute` is a structured operation similar to `linalg.generic` but with
    tile-typed block arguments. It iterates over a grid of tiles and applies
    the body computation to each tile position. Circular buffers are explicit
    operands: `input_cbs`/`output_cbs` align 1:1 with inputs/outputs (reusing
    the same CB value when the same tensor accessor is used multiple times).

    The operation takes tensors of tiles as inputs and outputs. The block
    arguments are individual tiles (`!ttcore.tile<H, W, dtype>`) extracted from
    the input/output tensors at each iteration point.

    This operation implements `TilingInterface` to support DST register capacity
    tiling via the transform dialect.

    In the pipeline, `ttl.compute` is lowered to `scf.for` loops by
    `ttl-lower-to-loops` BEFORE bufferization. The `ttl.tile_*` ops in the
    loop body are then converted to TTKernel ops after bufferization.

    Example:
    ```mlir
    %result = ttl.compute
        ins(%a, %b : tensor<4x4x!ttcore.tile<32x32, f32>>,
                     tensor<4x4x!ttcore.tile<32x32, f32>>)
        outs(%init : tensor<4x4x!ttcore.tile<32x32, f32>>) {
      ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
           %b_tile: !ttcore.tile<32x32, f32>,
           %out_tile: !ttcore.tile<32x32, f32>):
        %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
        ttl.yield %sum : !ttcore.tile<32x32, f32>
    } -> tensor<4x4x!ttcore.tile<32x32, f32>>
    ```
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    Variadic<TTL_CircularBuffer>:$input_cbs,
    Variadic<AnyRankedTensor>:$outputs,
    Variadic<TTL_CircularBuffer>:$output_cbs,
    AffineMapArrayAttr:$indexing_maps,
    ArrayAttr:$iterator_types
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // Return the number of inputs.
    unsigned getNumInputs() { return getInputs().size(); }

    // Return the number of outputs.
    unsigned getNumOutputs() { return getOutputs().size(); }
  }];
}

def TTL_YieldOp : TTL_Op<"yield", [ReturnLike, Terminator,
    ParentOneOf<["ComputeOp"]>]> {
  let summary = "Yield values from ttl.compute region";
  let description = [{
    `ttl.yield` terminates the body of `ttl.compute` and yields the computed
    tile values to be written to the output tensors.
  }];
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}

//===----------------------------------------------------------------------===//
// Tile-Level Operations (for use inside ttl.compute body)
//===----------------------------------------------------------------------===//

// Base class for tile operations
class TTL_TileOp<string mnemonic, list<Trait> traits = []>
    : TTL_Op<mnemonic, !listconcat(traits, [Pure])>;

class TTL_TileBinaryOp<string mnemonic, list<Trait> traits = []>
    : TTL_TileOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])> {
  let summary = "Binary tile operation";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

class TTL_TileUnaryOp<string mnemonic, list<Trait> traits = []>
    : TTL_TileOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])> {
  let summary = "Unary tile operation";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Binary Elementwise Operations (Tensor + Tile Ops)
//===----------------------------------------------------------------------===//

// Multiclass for defining paired tensor + tile binary ops
multiclass TTL_BinaryElementwisePair<string mnemonic, string kernelOp> {
  // Tensor-level op (operates on full tensors)
  def Op : TTL_ElementWiseBinary<mnemonic> {
    let description = "Elementwise "#mnemonic#" on tensors. Lowered to ttl.compute with ttl.tile_"#mnemonic#".";
  }
  // Tile-level op (operates inside ttl.compute body)
  def TileOp : TTL_TileBinaryOp<"tile_"#mnemonic> {
    let description = "Tile-level "#mnemonic#". Maps to ttkernel."#kernelOp#".";
  }
}

// Define all binary elementwise ops using the multiclass
defm TTL_Add : TTL_BinaryElementwisePair<"add", "add_tiles">;
defm TTL_Sub : TTL_BinaryElementwisePair<"sub", "sub_tiles">;
defm TTL_Mul : TTL_BinaryElementwisePair<"mul", "mul_tiles">;
defm TTL_Max : TTL_BinaryElementwisePair<"max", "max_tiles">;


//===----------------------------------------------------------------------===//
// Unary Elementwise Operations (Tensor + Tile Ops)
//===----------------------------------------------------------------------===//

// Multiclass for defining paired tensor + tile unary ops
multiclass TTL_UnaryElementwisePair<string mnemonic, string kernelOp> {
  // Tensor-level op (operates on full tensors)
  def Op : TTL_ElementWiseUnary<mnemonic> {
    let description = "Elementwise "#mnemonic#" on tensors. Lowered to ttl.compute with ttl.tile_"#mnemonic#".";
  }
  // Tile-level op (operates inside ttl.compute body)
  def TileOp : TTL_TileUnaryOp<"tile_"#mnemonic> {
    let description = "Tile-level "#mnemonic#". Maps to ttkernel."#kernelOp#".";
  }
}

// Define all unary elementwise ops using the multiclass
defm TTL_Exp : TTL_UnaryElementwisePair<"exp", "exp_tile">;
defm TTL_Log : TTL_UnaryElementwisePair<"log", "log_tile">;
defm TTL_Sqrt : TTL_UnaryElementwisePair<"sqrt", "sqrt_tile">;
defm TTL_Rsqrt : TTL_UnaryElementwisePair<"rsqrt", "rsqrt_tile">;
defm TTL_Tanh : TTL_UnaryElementwisePair<"tanh", "tanh_tile">;
defm TTL_Sigmoid : TTL_UnaryElementwisePair<"sigmoid", "sigmoid_tile">;
defm TTL_Neg : TTL_UnaryElementwisePair<"neg", "neg_tile">;
defm TTL_Abs : TTL_UnaryElementwisePair<"abs", "abs_tile">;
defm TTL_Relu : TTL_UnaryElementwisePair<"relu", "relu_tile">;

// Circular buffer synchronization operations
//===----------------------------------------------------------------------===//

def TTL_CBReserveOp : TTL_Op<"cb_reserve", [MemoryEffects<[MemWrite]>]> {
  let summary = "Reserve pages in a circular buffer for writing";
  let description = [{
    `ttl.cb_reserve` blocks until pages are available in the circular buffer
    for writing. Returns a tensor view of the reserved pages.

    This operation is used by producer threads (typically data movement kernels
    reading from DRAM) to acquire space in the CB before writing data.

    The result tensor has shape matching the CB's block shape and element type.
    The number of pages is derived from the CB's shape (elements per block).

    Example:
    ```mlir
    %view = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ```
  }];
  let arguments = (ins
    TTL_CircularBuffer:$cb
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$cb attr-dict `:` type($cb) `->` type($result)";
  let hasVerifier = 1;
}

def TTL_CBPushOp : TTL_Op<"cb_push", [MemoryEffects<[MemWrite]>]> {
  let summary = "Signal that pages have been written to the circular buffer";
  let description = [{
    `ttl.cb_push` signals that pages have been written to the circular buffer
    and are ready for consumer threads to read.

    This operation must be called after writing data to pages acquired via
    `ttl.cb_reserve`. It increments the CB's producer pointer.

    The number of pages is derived from the CB's shape (elements per block).

    Example:
    ```mlir
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ```
  }];
  let arguments = (ins
    TTL_CircularBuffer:$cb
  );
  let assemblyFormat = "$cb attr-dict `:` type($cb)";
  let hasVerifier = 1;
}

def TTL_CBWaitOp : TTL_Op<"cb_wait", [MemoryEffects<[MemRead]>]> {
  let summary = "Wait for pages to be available in the circular buffer";
  let description = [{
    `ttl.cb_wait` blocks until pages are available in the circular buffer for
    reading. Returns a tensor view of the available pages.

    This operation is used by consumer threads (typically compute kernels or
    data movement kernels writing to DRAM) to wait for data from producers.

    The number of pages is derived from the CB's shape (elements per block).

    Example:
    ```mlir
    %view = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ```
  }];
  let arguments = (ins
    TTL_CircularBuffer:$cb
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$cb attr-dict `:` type($cb) `->` type($result)";
  let hasVerifier = 1;
}

def TTL_CBPopOp : TTL_Op<"cb_pop", [MemoryEffects<[MemWrite]>]> {
  let summary = "Release pages from the circular buffer after reading";
  let description = [{
    `ttl.cb_pop` signals that pages have been consumed from the circular buffer
    and the space can be reused by producers.

    This operation must be called after reading data acquired via `ttl.cb_wait`.
    It increments the CB's consumer pointer.

    The number of pages is derived from the CB's shape (elements per block).

    Example:
    ```mlir
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ```
  }];
  let arguments = (ins
    TTL_CircularBuffer:$cb
  );
  let assemblyFormat = "$cb attr-dict `:` type($cb)";
  let hasVerifier = 1;
}

#endif // TTLANG_DIALECT_TTL_IR_TTLOPS_TD
