// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPS_TD
#define TTLANG_DIALECT_TTL_IR_TTLOPS_TD

include "ttlang/Dialect/TTL/IR/TTLBase.td"
include "ttlang/Dialect/TTL/IR/TTLOpsTypes.td"
include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TTL operation definitions
//===----------------------------------------------------------------------===//

def TTL_BindCBOp : TTL_Op<"bind_cb", [Pure]> {
  let summary = "Bind to a hardware circular buffer slot";
  let description = [{
    Declares that this program will use a hardware circular buffer slot and
    produces an SSA handle (`!ttl.cb`) representing that binding. This op does
    **not** allocate a buffer; the runtime/host is responsible for provisioning
    the CB. Binding occurs at kernel scope and the same SSA value should be
    threaded into data movement and compute threads.

    - `cb_index` is the circular buffer hardware slot index (0-31).
    - `buffer_factor` is the number of blocks (default 2 for double buffering).
    Example:

    ```mlir
    %cb = ttl.bind_cb {cb_index = 0 : index, buffer_factor = 2}
          : !ttl.cb<[1, 1], f32, 2>
    ```
  }];
  let arguments = (ins
    IndexAttr:$cb_index,
    DefaultValuedAttr<I64Attr, "2">:$buffer_factor
  );
  let results = (outs TTL_CircularBuffer:$result);
  let assemblyFormat = "`{` `cb_index` `=` $cb_index `,` `buffer_factor` `=` $buffer_factor `}` attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def TTL_AttachCBOp : TTL_Op<"attach_cb", []> {
  let summary = "Associate a tensor SSA value with a circular buffer";
  let description = [{
    Associates a tensor value with a circular buffer handle for later
    data-movement/compute lowering. This op is an identity on the tensor value
    but records a mapping (tensor -> CB) that later passes query. Exactly one
    CB should be attached per tensor SSA value.

    Note: This op is NOT marked Pure to prevent CSE/DCE from eliminating
    semantically meaningful bindings.

    Example:
    ```mlir
    %t_attached = ttl.attach_cb %t, %cb
        : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], f32, 2>)
          -> tensor<4x4x!ttcore.tile<32x32, f32>>
    ```
  }];
  let arguments = (ins
    AnyRankedTensor:$tensor,
    TTL_CircularBuffer:$cb
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$tensor `,` $cb attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TTL_TensorSliceOp : TTL_Op<"tensor_slice", [Pure]> {
  let summary = "Create a view into a tensor at specific tile indices";
  let description = [{
    `ttl.tensor_slice` creates a view into a TTNN tensor at a specific tile
    position. The resulting tensor can be passed to `ttl.copy` to transfer
    tiles starting at the specified position.

    The indices are tile coordinates (not element coordinates). For a tensor
    with shape [64, 64] and tile size 32x32, valid indices are [0,0], [0,1],
    [1,0], and [1,1].

    The result tensor has a reduced shape matching the CB dimensions but
    retains the original tensor's layout encoding. The lowering traces back
    to the original tensor via the defining op.

    Example:
    ```mlir
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %slice = ttl.tensor_slice %tensor[%c0, %c1]
             : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #layout>
               -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
    ```
  }];
  let arguments = (ins
    TTNNTensor:$tensor,
    Index:$tile_row,
    Index:$tile_col
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$tensor `[` $tile_row `,` $tile_col `]` attr-dict `:` type($tensor) `->` type($result)";
}

def TTL_CopyOp : TTL_Op<"copy", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Asynchronous copy between tensor slice and circular buffer";
  let description = [{
    `ttl.copy` initiates an asynchronous transfer and returns a transfer handle
    (`!ttl.transfer_handle`) that can be synchronized with `ttl.wait`.

    This operation is non-blocking. The destination is not safe to use until a
    corresponding `ttl.wait` has completed.

    Exactly one operand must be a circular buffer (`!ttl.cb`). The other operand
    must be a ranked tensor (the result of a tensor slice) with TTNN layout
    encoding (transfers all tiles via loop).

    TODO(ttl): Add an optional TRID attribute (range 0..15) when TRID-specific
    ttkernel noc ops land in tt-mlir. Issue: #87.

    Example:

    ```mlir
    #dram = #ttnn.buffer_type<dram>
    #layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

    func.func @dma_single(%t: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
      %cb = ttl.bind_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
      // Create a slice at tile position (0, 0)
      %c0 = arith.constant 0 : index
      %slice = ttl.tensor_slice %t[%c0, %c0] : tensor<32x32xf32, #layout> -> tensor<32x32xf32, #layout>
      %xf = ttl.copy %slice, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
      ttl.wait %xf : !ttl.transfer_handle<read>
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
    DeclareOpInterfaceMethods<DestinationStyleOpInterface>
  ]> {
  let summary = "Structured tile-level compute operation";
  let description = [{
    `ttl.compute` is a structured operation similar to `linalg.generic` but with
    tile-typed block arguments. It iterates over a grid of tiles and applies
    the body computation to each tile position. Circular buffers are associated
    to tensor operands/results via `ttl.attach_cb` (one CB per tensor SSA
    value); `ttl.compute` itself only carries tensor operands.

    The operation takes tensors of tiles as inputs and outputs. The block
    arguments are individual tiles (`!ttcore.tile<H, W, dtype>`) extracted from
    the input/output tensors at each iteration point.

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
// DST register control operations
//===----------------------------------------------------------------------===//

def TTL_InitSFPUOp : TTL_Op<"init_sfpu",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Initialize SFPU unpacker/packer for CB data formats.";
  let description = [{
    Configures the unpacker and packer hardware for SFPU operations based
    on the data formats of the input and output circular buffers. Must be
    called once before SFPU computations. The icb (input CB) and ocb (output
    CB) provide data format information for the unpacker and packer
    respectively.

    Both operands must be circular buffer handles obtained from `ttl.bind_cb`.
  }];
  let arguments = (ins TTL_CircularBuffer:$icb, TTL_CircularBuffer:$ocb);
  let assemblyFormat = "`(` $icb `,` $ocb `)` attr-dict `:` type($icb) `,` type($ocb)";
}

def TTL_TileRegsAcquireOp : TTL_Op<"tile_regs_acquire",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Acquire the DST register lock.";
  let description = [{
    Acquire an exclusive lock on the DST register before issuing tile
    computations that write to DST. This mirrors the TTKernel
    `tile_regs_acquire` operation (preferred over the deprecated
    `acquire_dst`). The call is blocking and ensures mutual exclusion on the DST
    register bank.
  }];

  let assemblyFormat = "attr-dict";
}

def TTL_TileRegsCommitOp : TTL_Op<"tile_regs_commit",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Commit the DST register lock (MATH thread).";
  let description = [{
    Release the DST register lock held by the MATH thread, mirroring
    `ttkernel.tile_regs_commit`. This is blocking and must be paired with a
    preceding `tile_regs_acquire`.
  }];
  let assemblyFormat = "attr-dict";
}

def TTL_TileRegsWaitOp : TTL_Op<"tile_regs_wait",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Wait for DST register availability (PACK thread).";
  let description = [{
    Acquire the DST register lock for the PACK thread, blocking until the MATH
    thread commits. Mirrors `ttkernel.tile_regs_wait`.
  }];
  let assemblyFormat = "attr-dict";
}

def TTL_TileRegsReleaseOp : TTL_Op<"tile_regs_release",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Release DST register lock (PACK thread).";
  let description = [{
    Release the DST register lock held by the PACK thread, mirroring
    `ttkernel.tile_regs_release`. Must be paired with a preceding
    `tile_regs_wait`.
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Tile-Level Operations (for use inside ttl.compute body)
//===----------------------------------------------------------------------===//

// Base class for tile operations
class TTL_TileOp<string mnemonic, list<Trait> traits = []>
    : TTL_Op<mnemonic, !listconcat(traits, [Pure, TTL_TileComputeOpTrait])>;

def TTL_TileType : TypeConstraint<
    CPred<"llvm::isa<mlir::tt::ttcore::TileType>($_self)">,
    "ttcore.tile type">;

def TTL_LinearizedIndexOp : TTL_Op<"linearized_index", [Pure]> {
  let summary = "Placeholder for row-major linearized index computation";
  let description = [{
    Represents a linearized index that will be computed from iteration dimensions
    during loop lowering. The `index_map` specifies the row-major linearization
    formula (e.g., affine_map<(d0, d1) -> (d0 * cols + d1)>).

    This op has no operands because it references iteration dimensions that don't
    exist as SSA values until loops are materialized. During loop lowering, this
    op is replaced with affine.apply using concrete loop IVs.

    Example:
    ```mlir
    %idx = ttl.linearized_index affine_map<(d0, d1) -> (d0 * 2 + d1)> : index
    ttl.copy_tile %tile, %idx, %dst_idx : ...
    ```
  }];
  let arguments = (ins AffineMapAttr:$index_map);
  let results = (outs Index:$result);
  let assemblyFormat = "$index_map attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def TTL_CopyTileOp : TTL_Op<"copy_tile",
                            [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Copy a tile from a circular buffer into a DST register";
  let description = [{
    Performs a CB -> DST copy, matching `copy_tile_init` + `copy_tile` in the
    TTKernel API. The source is identified by a circular buffer operand and an
    input tile index; the destination is the DST register index. This is used by
    the DST assignment pass to make CB -> DST copies explicit before tile-level
    compute is lowered.
  }];
  let arguments = (ins
    TTL_TileType:$src,
    Index:$src_index,
    Index:$dst_index
  );
  let results = (outs TTL_DSTRegister:$dst_token, TTL_TileType:$dst_tile);
  let assemblyFormat =
      "$src `,` $src_index `,` $dst_index attr-dict `:` "
      "type($src) `,` type($src_index) `,` type($dst_index) "
      "`->` type($dst_token) `,` type($dst_tile)";
  let hasVerifier = 1;
}

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

def TTL_CBReserveOp : TTL_Op<"cb_reserve",
    [MemoryEffects<[MemWrite]>, DeclareOpInterfaceMethods<ViewLikeOpInterface>]> {
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

def TTL_CBWaitOp : TTL_Op<"cb_wait",
    [MemoryEffects<[MemRead]>, DeclareOpInterfaceMethods<ViewLikeOpInterface>]> {
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

def TTL_StoreOp : TTL_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let summary = "Store a tile into a circular buffer view";
  let description = [{
    `ttl.store` packs a single tile from the DST register into a circular buffer
    view obtained from `ttl.cb_reserve`. This operation is used by compute
    kernels to write computed results into output circular buffers.

    The tile value typically comes from compute operations (e.g., tile_add,
    tile_mul). The view must be from a prior `ttl.cb_reserve` call on the
    same CB.

    Example:
    ```mlir
    %view = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
                                 -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %tile, %view : !ttcore.tile<32x32, bf16>,
                             tensor<1x1x!ttcore.tile<32x32, bf16>>
    ```
  }];
  let arguments = (ins
    AnyType:$tile,
    AnyRankedTensor:$view
  );
  let assemblyFormat = "$tile `,` $view attr-dict `:` type($tile) `,` type($view)";
  let hasVerifier = 1;
}

#endif // TTLANG_DIALECT_TTL_IR_TTLOPS_TD
