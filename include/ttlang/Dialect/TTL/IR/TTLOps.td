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

//===----------------------------------------------------------------------===//
// Binary Elementwise Operations
//===----------------------------------------------------------------------===//

def TTL_AddOp : TTL_ElementWiseBinary<"add"> {
  let description = [{
    Elementwise addition on tensors. Lowered via a shared elementwise pathway
    to linalg.generic, tiled for DST capacity, then to TTKernel tile kernels.
  }];
}

def TTL_SubOp : TTL_ElementWiseBinary<"sub"> {
  let description = [{
    Elementwise subtraction on tensors using the shared elementwise pathway.
  }];
}

def TTL_MulOp : TTL_ElementWiseBinary<"mul"> {
  let description = [{
    Elementwise multiplication on tensors using the shared elementwise pathway.
  }];
}

def TTL_MaxOp : TTL_ElementWiseBinary<"max"> {
  let description = [{
    Elementwise maximum of two tensors using the shared elementwise pathway.
  }];
}

//===----------------------------------------------------------------------===//
// Unary Elementwise Operations (SFPU Path)
//===----------------------------------------------------------------------===//

def TTL_ExpOp : TTL_ElementWiseUnary<"exp"> {
  let description = [{
    Elementwise exponential using the SFPU path in TTKernel. Lowered through
    the shared elementwise pipeline.
  }];
}

def TTL_LogOp : TTL_ElementWiseUnary<"log"> {
  let description = [{
    Elementwise natural logarithm using the SFPU path in TTKernel.
  }];
}

def TTL_SqrtOp : TTL_ElementWiseUnary<"sqrt"> {
  let description = [{
    Elementwise square root using the SFPU path in TTKernel.
  }];
}

def TTL_RsqrtOp : TTL_ElementWiseUnary<"rsqrt"> {
  let description = [{
    Elementwise reciprocal square root (1/sqrt(x)) using the SFPU path.
  }];
}

def TTL_TanhOp : TTL_ElementWiseUnary<"tanh"> {
  let description = [{
    Elementwise hyperbolic tangent using the SFPU path in TTKernel.
  }];
}

def TTL_SigmoidOp : TTL_ElementWiseUnary<"sigmoid"> {
  let description = [{
    Elementwise sigmoid (1/(1+exp(-x))) using the SFPU path in TTKernel.
  }];
}

def TTL_NegOp : TTL_ElementWiseUnary<"neg"> {
  let description = [{
    Elementwise negation using the shared elementwise pathway.
  }];
}

def TTL_AbsOp : TTL_ElementWiseUnary<"abs"> {
  let description = [{
    Elementwise absolute value using the shared elementwise pathway.
  }];
}

def TTL_ReluOp : TTL_ElementWiseUnary<"relu"> {
  let description = [{
    Elementwise ReLU (max(x, 0)) using the SFPU path in TTKernel. Lowered
    through the shared elementwise pipeline.
  }];
}

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
    the body computation to each tile position.

    The operation takes tensors of tiles as inputs and outputs. The block
    arguments are individual tiles (`!ttcore.tile<H, W, dtype>`) extracted from
    the input/output tensors at each iteration point.

    This operation implements `TilingInterface` to support DST register capacity
    tiling via the transform dialect.

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
    Variadic<AnyRankedTensor>:$outputs,
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

    // Return the iteration space rank.
    unsigned getNumLoops();

    // Return the block arguments corresponding to inputs.
    Block::BlockArgListType getInputBlockArguments();

    // Return the block arguments corresponding to outputs.
    Block::BlockArgListType getOutputBlockArguments();
  }];
}

def TTL_YieldOp : TTL_Op<"yield", [Pure, ReturnLike, Terminator,
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

// Binary tile operations
def TTL_TileAddOp : TTL_TileBinaryOp<"tile_add"> {
  let description = [{ Tile-level addition. Maps to ttkernel.add_tiles. }];
}

def TTL_TileSubOp : TTL_TileBinaryOp<"tile_sub"> {
  let description = [{ Tile-level subtraction. Maps to ttkernel.sub_tiles. }];
}

def TTL_TileMulOp : TTL_TileBinaryOp<"tile_mul"> {
  let description = [{ Tile-level multiplication. Maps to ttkernel.mul_tiles. }];
}

def TTL_TileMaxOp : TTL_TileBinaryOp<"tile_max"> {
  let description = [{ Tile-level maximum. }];
}

// Unary tile operations (SFPU path)
def TTL_TileExpOp : TTL_TileUnaryOp<"tile_exp"> {
  let description = [{ Tile-level exponential. Maps to ttkernel.exp_tile. }];
}

def TTL_TileLogOp : TTL_TileUnaryOp<"tile_log"> {
  let description = [{ Tile-level natural logarithm. Maps to ttkernel.log_tile. }];
}

def TTL_TileSqrtOp : TTL_TileUnaryOp<"tile_sqrt"> {
  let description = [{ Tile-level square root. Maps to ttkernel.sqrt_tile. }];
}

def TTL_TileRsqrtOp : TTL_TileUnaryOp<"tile_rsqrt"> {
  let description = [{ Tile-level reciprocal square root. }];
}

def TTL_TileTanhOp : TTL_TileUnaryOp<"tile_tanh"> {
  let description = [{ Tile-level hyperbolic tangent. Maps to ttkernel.tanh_tile. }];
}

def TTL_TileSigmoidOp : TTL_TileUnaryOp<"tile_sigmoid"> {
  let description = [{ Tile-level sigmoid. }];
}

def TTL_TileNegOp : TTL_TileUnaryOp<"tile_neg"> {
  let description = [{ Tile-level negation. }];
}

def TTL_TileAbsOp : TTL_TileUnaryOp<"tile_abs"> {
  let description = [{ Tile-level absolute value. }];
}

def TTL_TileReluOp : TTL_TileUnaryOp<"tile_relu"> {
  let description = [{ Tile-level ReLU. Maps to ttkernel.relu_tile. }];
}

// DST register markers (analysis only; removed by lowering passes).
def TTL_AcquireDSTOp : TTL_Op<"acquire_dst", [Pure]> {
  let results = (outs Index:$dst_reg);
  let summary = "Acquire a DST register (analysis marker)";
  let description = [{
    Analysis marker indicating a DST register acquisition. Removed after
    register assignment.
  }];
  let assemblyFormat = "attr-dict";
}

def TTL_ReleaseDSTOp : TTL_Op<"release_dst", [Pure]> {
  let arguments = (ins Index:$dst_reg);
  let summary = "Release a DST register (analysis marker)";
  let description = [{
    Analysis marker indicating a DST register release. Removed after register
    assignment.
  }];
  let assemblyFormat = "$dst_reg attr-dict";
}

def TTL_RequireDSTOp : TTL_Op<"require_dst", [Pure]> {
  let arguments = (ins AnyType:$value);
  let summary = "Assert value should reside in DST";
  let description = [{
    Hint used by register assignment to keep a value in DST registers.
  }];
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

#endif // TTLANG_DIALECT_TTL_IR_TTLOPS_TD
