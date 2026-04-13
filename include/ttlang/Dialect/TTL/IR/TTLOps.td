// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPS_TD
#define TTLANG_DIALECT_TTL_IR_TTLOPS_TD

include "ttlang/Dialect/TTL/IR/TTLBase.td"
include "ttlang/Dialect/TTL/IR/TTLOpsTypes.td"
include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td"
include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/IndexingMapOpInterface.td"
include "mlir/Interfaces/TilingInterface.td"
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
    - `block_count` is the number of blocks (default 2 for double buffering).
    Example:

    ```mlir
    %cb = ttl.bind_cb {cb_index = 0 : index, block_count = 2}
          : !ttl.cb<[1, 1], f32, 2>
    ```
  }];
  let arguments = (ins
    IndexAttr:$cb_index,
    DefaultValuedAttr<I64Attr, "2">:$block_count
  );
  let results = (outs TTL_CircularBuffer:$result);
  let assemblyFormat = "`{` `cb_index` `=` $cb_index `,` `block_count` `=` $block_count `}` attr-dict `:` type($result)";
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
    `ttl.tensor_slice` creates a view into a tensor at a specific tile
    position. The resulting tensor can be passed to `ttl.copy` to transfer
    tiles starting at the specified position.

    The indices are tile coordinates (not element coordinates). The number
    of indices must match the rank of the tensor's tile grid. For a 2D
    tensor with shape [64, 64] and tile size 32x32, valid indices are
    [0,0], [0,1], [1,0], and [1,1]. For an ND tensor, indices cover all
    dimensions of the tile grid.

    The result tensor has a reduced shape matching the CB dimensions but
    retains the original tensor's layout encoding. The lowering traces back
    to the original tensor via the defining op.

    Example:
    ```mlir
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %slice = ttl.tensor_slice %tensor[%c0, %c1]
             : tensor<2x2x!ttcore.tile<32x32, bf16>, #layout>
               -> tensor<1x1x!ttcore.tile<32x32, bf16>, #layout>
    ```
  }];
  let arguments = (ins
    TTLLayoutTensor:$tensor,
    Variadic<Index>:$indices
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$tensor `[` $indices `]` attr-dict `:` type($tensor) `->` type($result)";
  let hasVerifier = 1;
}

def TTL_CopyOp : TTL_Op<"copy", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Asynchronous copy between tensor slice and circular buffer";
  let description = [{
    `ttl.copy` initiates an asynchronous transfer and returns a transfer handle
    (`!ttl.transfer_handle`) that can be synchronized with `ttl.wait`.

    This operation is non-blocking. The destination is not safe to use until a
    corresponding `ttl.wait` has completed.

    Exactly one operand must be a circular buffer (`!ttl.cb`). The other operand
    must be a ranked tensor (the result of a tensor slice) with TTL layout
    encoding (transfers all tiles via loop).

    TODO(ttl): Add an optional TRID attribute (range 0..15) when TRID-specific
    ttkernel noc ops land in tt-mlir. Issue: #87.

    Example:

    ```mlir
    #layout = #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, f32>,
                          buffer = L1, grid = [1, 1], memory = Interleaved>

    func.func @dma_single(%t: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
      %cb = ttl.bind_cb() {shape = [1, 1], element_type = f32, block_count = 2} : !ttl.cb<[1, 1], f32, 2>
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
          !listconcat(traits, [Pure, TTL_BinaryElementwiseOpTrait])> {
  let summary = "Binary elementwise operation";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

class TTL_ElementWiseUnary<string mnemonic, list<Trait> traits = []>
    : TTL_ElementWiseBase<
          mnemonic,
          !listconcat(traits, [Pure, TTL_UnaryElementwiseOpTrait])> {
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
    DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
    IndexingMapOpInterface,
    DeclareOpInterfaceMethods<TilingInterface,
      ["getLoopIteratorTypes", "getIterationDomain",
       "getTiledImplementation", "getResultTilePosition"]>
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
        ttl.yield
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

    // Return the static shape of the operand's tensor type.
    // Required by IndexingMapOpInterface default implementations
    // (getStaticOperandShape, getStaticLoopRanges).
    ArrayRef<int64_t> getShape(OpOperand *opOperand) {
      auto shape = cast<RankedTensorType>(opOperand->get().getType()).getShape();
      return shape;
    }

    // Return iterator types as a vector of utils::IteratorType.
    SmallVector<utils::IteratorType> getIteratorTypesArray();

    // Return a flat list of all operand dimensions as OpFoldResults.
    SmallVector<OpFoldResult> createFlatListOfOperandDims(OpBuilder &b,
                                                          Location loc);

    // Extract static dimension sizes from the iteration domain.
    SmallVector<int64_t> getStaticIterationDomainSizes();

    // Compute total tiles in the iteration domain. Handles multi-output
    // computes with potentially different output shapes, as well as
    // reductions where the output shape is smaller than the iteration domain.
    int64_t getTotalIterationTiles();

    // Return whether the body contains an op of the given type.
    template <typename OpTy>
    bool containsOp() {
      bool found = false;
      getBody().walk([&](OpTy) { found = true; return WalkResult::interrupt(); });
      return found;
    }
  }];
}

def TTL_YieldOp : TTL_Op<"yield", [ReturnLike, Terminator,
    ParentOneOf<["ComputeOp", "DstSectionOp"]>]> {
  let summary = "Terminate a ttl.compute or ttl.dst_section region";
  let description = [{
    `ttl.yield` terminates the body of `ttl.compute` or `ttl.dst_section`.
  }];
  let assemblyFormat = "attr-dict";
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}

//===----------------------------------------------------------------------===//
// DST register section
//===----------------------------------------------------------------------===//

def TTL_DstSectionOp : TTL_Op<"dst_section", [
    SingleBlockImplicitTerminator<"YieldOp">,
    RecursiveMemoryEffects,
    NoRegionArguments
  ]> {
  let summary = "DST register acquisition scope";
  let description = [{
    Demarcates a DST register section: all tile compute ops and stores in
    the body share one acquire/release cycle. Lowered to:

      acquire -> [math ops] -> commit -> wait -> [pack ops] -> release

    The body must have math-phase ops before pack-phase ops (stores,
    cb_push). `ttl-lower-to-loops` enforces this ordering, including after
    subblock unrolling.

    Granularity: one per tile loop iteration (non-subblocked), one per
    unrolled subblock, or one per parallel iteration with the reduction
    loop inside (accumulating computes).
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";

  let builders = [
    OpBuilder<(ins), [{
      Region *bodyRegion = $_state.addRegion();
      Block *block = new Block();
      bodyRegion->push_back(block);
      OpBuilder::InsertionGuard guard($_builder);
      $_builder.setInsertionPointToEnd(block);
      YieldOp::create($_builder, $_state.location);
    }]>
  ];
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
// Core indexing operations (for multi-core kernels)
//===----------------------------------------------------------------------===//

def TTL_CoreXOp : TTL_Op<"core_x", [Pure]> {
  let summary = "Get the X coordinate of the current core";
  let description = [{
    Returns the X coordinate (column) of the core executing this kernel.
    For a grid of shape (rows, cols), X ranges from 0 to cols-1.

    Lowers to `ttkernel.my_x`.

    Example:
    ```mlir
    %x = ttl.core_x : index
    ```
  }];
  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def TTL_CoreYOp : TTL_Op<"core_y", [Pure]> {
  let summary = "Get the Y coordinate of the current core";
  let description = [{
    Returns the Y coordinate (row) of the core executing this kernel.
    For a grid of shape (rows, cols), Y ranges from 0 to rows-1.

    Lowers to `ttkernel.my_y`.

    Example:
    ```mlir
    %y = ttl.core_y : index
    ```
  }];
  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Profiling operations
//===----------------------------------------------------------------------===//

def TTL_SignpostOp : TTL_Op<"signpost", []> {
  let summary = "Profiling marker for device tracing";
  let description = [{
    Marks a profiling region with a custom name for Tracy profiler visualization.
    This operation generates DeviceZoneScopedN markers in the emitted C++ code.

    Example:
    ```mlir
    ttl.signpost "my_custom_region"
    ```
  }];

  let arguments = (ins StrAttr:$name, UnitAttr:$is_end);
  let assemblyFormat = "$name attr-dict";
}

//===----------------------------------------------------------------------===//
// Debug print operations
//===----------------------------------------------------------------------===//

def TTL_DPrintOp : TTL_Op<"dprint", []> {
  let summary = "Debug print from device kernel";
  let description = [{
    Emits device debug print calls. Mode determines the output format:
    - "scalar": prints format string with scalar values
    - "cb": prints CB metadata (size, limit, ptrs)
    - "tile": prints full 32x32 tile contents from a CB
    - "tensor": prints raw memory pages from tensor backing buffer
    - "dst": dumps live DST register slots

    Example:
    ```mlir
    ttl.dprint "x = {}" (%x) {mode = "scalar"} : (index)
    ttl.dprint "input" (%cb) {mode = "cb"} : (!ttl.cb<...>)
    ttl.dprint "after exp" () {mode = "dst"}
    ```
  }];

  let arguments = (ins
    StrAttr:$fmt,
    StrAttr:$mode,
    Variadic<AnyType>:$argv,
    OptionalAttr<StrAttr>:$thread,
    OptionalAttr<I64Attr>:$num_pages
  );

  let assemblyFormat = "$fmt `(` $argv `)` attr-dict `:` `(` type($argv) `)`";
}

//===----------------------------------------------------------------------===//
// Tile-Level Operations (for use inside ttl.compute body)
//===----------------------------------------------------------------------===//

// Base class for tile operations
// Note: These ops are NOT Pure because they write to DST registers.
class TTL_TileOp<string mnemonic, list<Trait> traits = []>
    : TTL_Op<mnemonic, !listconcat(traits, [TTL_TileComputeOpTrait])>;

def TTL_TileType : Type<
    CPred<"llvm::isa<mlir::tt::ttcore::TileType>($_self)">,
    "ttcore.tile type">;

// Tile type inside compute body; ranked tensor after lower-matmul-block.
def TTL_TileOrTensor : AnyTypeOf<[TTL_TileType, AnyRankedTensor]>;

def TTL_IterIndexOp : TTL_Op<"iter_index", [Pure,
    ParentOneOf<["ComputeOp"]>]> {
  let summary = "Get the current iteration index for a dimension";
  let description = [{
    Returns the current iteration coordinate for the given dimension of the
    enclosing `ttl.compute`'s iteration domain. Follows the `linalg.index`
    pattern.

    Created by `ttl-assign-dst` alongside `ttl.copy_tile` and
    `ttl.tile_store` ops. The per-operand indexing map is applied to the
    iter_index results (via `affine.apply`) to produce operand-space
    coordinates for `copy_tile` and `tile_store`. For identity maps, the
    iter_index values pass through unchanged.

    During `ttl-lower-to-loops`, each `ttl.iter_index` is replaced with the
    corresponding loop induction variable. During `ttl-subblock-compute-for-dst`,
    subblock offsets are added to the iter_index results for subblocked
    dimensions (same adjustment pattern as `linalg.index` during tiling).

    Example (identity map -- iter_index values used directly as indices):
    ```mlir
    ttl.compute ins(%in : tensor<2x3x!tile>) outs(%out : tensor<2x3x!tile>)
        {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>],
         iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a: !ttcore.tile<32x32, bf16>, %b: !ttcore.tile<32x32, bf16>):
        %i = ttl.iter_index 0 : index
        %j = ttl.iter_index 1 : index
        %tok, %tile = ttl.copy_tile %a[%i, %j] into dst[%c0] : ...
        ttl.tile_store %result, %view[%i, %j] : ...
        ttl.yield
    }
    ```
  }];
  let arguments = (ins I64Attr:$dim);
  let results = (outs Index:$result);
  let assemblyFormat = "$dim attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def TTL_CopyTileOp : TTL_Op<"copy_tile",
                            [MemoryEffects<[MemRead, MemWrite]>,
                             TTL_CBInputTileOpTrait,
                             TTL_DataMovementOpTrait,
                             TTL_DstResultOpTrait]> {
  let summary = "Copy a tile from a circular buffer into a DST register";
  let description = [{
    Performs a CB -> DST copy, matching `copy_tile_init` + `copy_tile` in the
    TTKernel API. The source is identified by a circular buffer operand and
    multi-dimensional CB tile coordinates; the destination is the DST register
    index.

    The `src_indices` are multi-dimensional CB tile coordinates derived from
    `ttl.iter_index` ops and the per-operand indexing map. They are populated
    by `ttl-assign-dst` inside the `ttl.compute` body and resolved to concrete
    loop IVs by `ttl-lower-to-loops`. At `convert-ttl-to-ttkernel` time, the
    indices are linearized to a flat CB tile index for `copy_tile`.

    Example:
    ```mlir
    // After assign-dst (inside ttl.compute body, iter_index provides coordinates):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %tok, %tile = ttl.copy_tile %src[%i, %j] into dst[%c0] : !ttcore.tile<32x32, bf16>
                  -> !ttl.dst, !ttcore.tile<32x32, bf16>
    // After lower-to-loops (iter_index replaced with loop IVs):
    %tok, %tile = ttl.copy_tile %src[%row, %col] into dst[%c0] : !ttcore.tile<32x32, bf16>
                  -> !ttl.dst, !ttcore.tile<32x32, bf16>
    ```
  }];
  let arguments = (ins
    TTL_TileType:$src,
    Variadic<Index>:$src_indices,
    Index:$dst_index
  );
  let results = (outs TTL_DSTRegister:$dst_token, TTL_TileType:$dst_tile);
  let assemblyFormat =
      "$src `[` $src_indices `]` `into` `dst` `[` $dst_index `]` attr-dict `:` "
      "type($src) "
      "`->` type($dst_token) `,` type($dst_tile)";
  let hasVerifier = 1;
}

def TTL_CopyDstOp : TTL_Op<"copy_dst",
                           [TTL_DstResultOpTrait,
                            AllTypesMatch<["src_tile", "result"]>,
                            MemoryEffects<[MemRead, MemWrite]>,
                            TTL_DSTInputsTrait,
                            TTL_DataMovementOpTrait]> {
  let summary = "Copy a tile from one DST register to another";
  let description = [{
    Performs DST -> DST copy for multi-consumer value preservation. Used when
    a value has multiple consumers and at least one is unary (which overwrites
    in-place). The copy ensures earlier consumers work on independent data.

    The dst_index operand specifies the destination DST register. The source
    register is determined by the src_tile's dst_index.

    Lowering: ttl.copy_dst -> ttkernel.copy_dest_values(src.dst_index, dst_index)

    Example:
    ```mlir
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = ttl.tile_mul %a, %b into dst[%c2] : !ttcore.tile<32x32, f32>
    %0_copy = ttl.copy_dst %0 into dst[%c3] : !ttcore.tile<32x32, f32>
    %1 = ttl.tile_abs %0_copy into dst[%c3] : !ttcore.tile<32x32, f32>
    %2 = ttl.tile_exp %0 into dst[%c2] : !ttcore.tile<32x32, f32>
    ```
  }];
  let arguments = (ins TTL_TileType:$src_tile, Index:$dst_index);
  let results = (outs TTL_TileType:$result);
  let assemblyFormat = "$src_tile `into` `dst` `[` $dst_index `]` attr-dict `:` type($src_tile) `->` type($result)";
}

// Note: DSTInputsTrait is the default for binary ops. FPU-marked binaries
// (ttl.fpu_binary) read from CB instead; use isCBInputOp() to check at runtime.
class TTL_TileBinaryOp<string mnemonic, list<Trait> traits = []>
    : TTL_TileOp<mnemonic, !listconcat(traits, [TTL_TileBinaryOpTrait,
                                                 TTL_DSTInputsTrait,
                                                 TTL_DstResultOpTrait,
                                                 AllTypesMatch<["lhs", "rhs", "result"]>])> {
  let summary = "Binary tile operation";
  let arguments = (ins TTL_TileType:$lhs, TTL_TileType:$rhs, Index:$dst_index);
  let results = (outs TTL_TileType:$result);
  let assemblyFormat = "$lhs `,` $rhs `into` `dst` `[` $dst_index `]` attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

class TTL_TileUnaryOp<string mnemonic, list<Trait> traits = []>
    : TTL_TileOp<mnemonic, !listconcat(traits, [TTL_TileUnaryOpTrait,
                                                 TTL_DSTInputsTrait, TTL_InPlaceOpTrait,
                                                 TTL_DstResultOpTrait,
                                                 AllTypesMatch<["input", "result"]>])> {
  let summary = "Unary tile operation";
  let arguments = (ins TTL_TileType:$input, Index:$dst_index);
  let results = (outs TTL_TileType:$result);
  let assemblyFormat = "$input `into` `dst` `[` $dst_index `]` attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Binary Elementwise Operations (Tensor + Tile Ops)
//===----------------------------------------------------------------------===//

// Multiclass for defining paired tensor + tile binary ops
multiclass TTL_BinaryElementwisePair<string mnemonic, string kernelOp,
                                     list<Trait> extraTraits = []> {
  // Tensor-level op (operates on full tensors)
  def Op : TTL_ElementWiseBinary<mnemonic> {
    let description = "Elementwise "#mnemonic#" on tensors. Lowered to ttl.compute with ttl.tile_"#mnemonic#".";
  }
  // Tile-level op (operates inside ttl.compute body)
  def TileOp : TTL_TileBinaryOp<"tile_"#mnemonic, extraTraits> {
    let description = "Tile-level "#mnemonic#". Maps to ttkernel."#kernelOp#".";
  }
}

// Define all binary elementwise ops using the multiclass
defm TTL_Add : TTL_BinaryElementwisePair<"add", "add_tiles">;
defm TTL_Sub : TTL_BinaryElementwisePair<"sub", "sub_tiles">;
defm TTL_Mul : TTL_BinaryElementwisePair<"mul", "mul_tiles">;
defm TTL_Div : TTL_BinaryElementwisePair<"div", "div_binary_tile">;
defm TTL_Max : TTL_BinaryElementwisePair<"max", "max_tiles",
                                         [TTL_InPlaceOpTrait]>;
defm TTL_Min : TTL_BinaryElementwisePair<"min", "binary_min_tile",
                                         [TTL_InPlaceOpTrait]>;


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
defm TTL_Floor : TTL_UnaryElementwisePair<"floor", "floor_tile">;
defm TTL_Recip : TTL_UnaryElementwisePair<"recip", "recip_tile">;
defm TTL_Sin : TTL_UnaryElementwisePair<"sin", "sin_tile">;
defm TTL_Cos : TTL_UnaryElementwisePair<"cos", "cos_tile">;
defm TTL_Tan : TTL_UnaryElementwisePair<"tan", "tan_tile">;
defm TTL_Asin : TTL_UnaryElementwisePair<"asin", "asin_tile">;
defm TTL_Acos : TTL_UnaryElementwisePair<"acos", "acos_tile">;
defm TTL_Atan : TTL_UnaryElementwisePair<"atan", "atan_tile">;

//===----------------------------------------------------------------------===//
// Broadcast Operation
//===----------------------------------------------------------------------===//

def TTL_BcastOp : TTL_Op<"bcast", []> {
  let summary = "Broadcast a row/column/scalar tensor to full tiles";
  let description = [{
    Broadcasts the input tensor to full tiles based on bcast_type.
    Reads from CB, writes result to DST then packs to output CB.
    Both input and output must be CB-attached tensors.

    bcast_type specifies the broadcast dimension:
    - Row: broadcast first row to all rows
    - Col: broadcast first column to all columns
    - Scalar: broadcast single value to all elements
  }];
  let arguments = (ins AnyType:$input, AnyType:$output, TTL_BcastTypeAttr:$bcast_type);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `,` $output $bcast_type attr-dict `:` `(` type($input) `,` type($output) `)` `->` type($result)";
}

def TTL_TileBcastOp : TTL_TileOp<"tile_bcast", [TTL_CBInputTileOpTrait, TTL_DstResultOpTrait]> {
  let summary = "Tile-level broadcast operation";
  let description = [{
    Broadcasts a row, column, or scalar tile into a full 32x32 tile.
    Maps to ttkernel.unary_bcast_init + ttkernel.unary_bcast.
    Reads input from CB, writes result to DST. The output operand carries
    the output CB for init-time PACK configuration only.
  }];
  let arguments = (ins AnyType:$input, AnyType:$output, TTL_BcastTypeAttr:$bcast_type, Index:$dst_index);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `,` $output $bcast_type `into` `dst` `[` $dst_index `]` attr-dict `:` `(` type($input) `,` type($output) `)` `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Reduce Operation
//===----------------------------------------------------------------------===//

def TTL_ReduceOp : TTL_Op<"reduce", [Pure]> {
  let summary = "Reduce a tensor along specified dimensions with a scaling factor";
  let description = [{
    Performs a scaled reduction (sum or max) along the specified dimensions.
    The result tensor has the reduced dimensions collapsed to size 1.
    The scaler tensor is multiplied into every result tile.

    `dims` specifies which dimensions to reduce, using Python-style indexing
    (negative values count from the end). For a rank-2 input:
    - dims=[-2] or dims=[0]: reduce rows
    - dims=[-1] or dims=[1]: reduce columns
    - dims=[-2,-1] or dims=[0,1]: reduce both

    Both input and scaler must be CB-attached rank-2 tensors.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$scaler,
                       TTL_ReduceTypeAttr:$reduce_type,
                       DenseI64ArrayAttr:$dims);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input `,` $scaler $reduce_type $dims attr-dict `:`
    `(` type($input) `,` type($scaler) `)` `->` type($result)
  }];
  let hasVerifier = 1;
}

def TTL_TileReduceOp : TTL_TileOp<"tile_reduce",
    [TTL_CBInputTileOpTrait, TTL_AccumulatingOpTrait, TTL_DstResultOpTrait]> {
  let summary = "Tile-level reduction with scaling";
  let description = [{
    Reduces a tile from the input dataflow buffer using the specified
    reduction function (sum or max), scaled by a tile from the scaler
    dataflow buffer, and writes the result to DST.
  }];
  let arguments = (ins AnyType:$input, AnyType:$scaler, AnyType:$output,
                       TTL_ReduceTypeAttr:$reduce_type,
                       TTKernel_ReduceDimAttr:$reduce_dim,
                       Index:$dst_index);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $input `,` $scaler `,` $output $reduce_type $reduce_dim `into` `dst` `[` $dst_index `]` attr-dict `:`
    `(` type($input) `,` type($scaler) `,` type($output) `)` `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Transpose Operation
//===----------------------------------------------------------------------===//

def TTL_TransposeOp : TTL_Op<"transpose", [Pure]> {
  let summary = "Transpose a 2D tensor of tiles";
  let description = [{
    Transposes a rank-2 tensor: shape (M, N) becomes (N, M).
    Both the tile grid and individual 32x32 tiles are transposed.
    Input must be a CB-attached rank-2 tensor.
  }];
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def TTL_TileTransposeOp : TTL_TileOp<"tile_transpose", [TTL_CBInputTileOpTrait, TTL_DstResultOpTrait]> {
  let summary = "Tile-level 32x32 transpose (dataflow buffer input, DST output)";
  let description = [{
    Transposes a 32x32 tile from a dataflow buffer and writes the result
    to a DST register.
  }];
  let arguments = (ins AnyType:$input, AnyType:$output, Index:$dst_index);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $input `,` $output `into` `dst` `[` $dst_index `]` attr-dict `:` `(` type($input) `,` type($output) `)` `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Fill Operation
//===----------------------------------------------------------------------===//

def TTL_FillOp : TTL_Op<"fill", [Pure]> {
  let summary = "Fill a tensor with a constant value";
  let description = [{
    Fills all elements of a tensor with the given f32 value.
    The output CB is determined by the ttl.store on this op's result.
  }];
  let arguments = (ins F32Attr:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def TTL_TileFillOp : TTL_TileOp<"tile_fill", [TTL_DstResultOpTrait]> {
  let summary = "Tile-level fill operation";
  let description = [{
    Fills a DST register tile with a constant f32 value.
    Maps to ttkernel.fill_tile_init + ttkernel.fill_tile.
  }];
  let arguments = (ins F32Attr:$value, Index:$dst_index);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$value `into` `dst` `[` $dst_index `]` attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Matrix Multiplication
//===----------------------------------------------------------------------===//

def TTL_MatmulOp : TTL_Op<"matmul", [Pure]> {
  let summary = "Matrix multiply two tensors of tiles";
  let description = [{
    Block-level matrix multiply: result[i, j] = sum_k(lhs[i, k] * rhs[k, j]).
    Both inputs must be rank-2 tensors of tiles. The inner (K) dimensions
    must match: lhs.shape[1] == rhs.shape[0]. Result shape is
    [lhs.shape[0], rhs.shape[1]].

    Lowered to ttl.compute with ttl.tile_matmul_block by convert-ttl-to-compute.
  }];
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

def TTL_TileMatmulBlockOp : TTL_TileOp<"tile_matmul_block",
    [TTL_CBInputTileOpTrait, TTL_DstResultOpTrait]> {
  let summary = "Block-level matrix multiply-accumulate (CB inputs, DST output)";
  let description = [{
    Multiplies a block of A tiles by a block of B tiles and accumulates
    into DST registers (DST += A * B). Both A and B are read from circular
    buffers. The block dimensions (rt, ct, kt) are derived from the
    enclosing ttl.compute's operand shapes during TTKernel lowering.

    When the optional `accumulator` operand is present, it is pre-loaded
    into DST via copy_tile before the matmul. The result is
    accumulator + lhs @ rhs, exploiting matmul_block's additive
    accumulation to eliminate an explicit add.

    Maps to ttkernel.mm_block_init + ttkernel.experimental::matmul_block.
  }];
  let arguments = (ins TTL_TileOrTensor:$lhs, TTL_TileOrTensor:$rhs,
                       Optional<TTL_TileOrTensor>:$accumulator,
                       Index:$dst_index);
  let results = (outs TTL_TileOrTensor:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs (`,` $accumulator^)? `into` `dst` `[` $dst_index `]` attr-dict `:`
    type($lhs) `,` type($rhs) (`,` type($accumulator)^)? `->` type($result)
  }];
}

// Circular buffer synchronization operations
//===----------------------------------------------------------------------===//

def TTL_CBReserveOp : TTL_Op<"cb_reserve",
    [MemoryEffects<[MemWrite]>, DeclareOpInterfaceMethods<ViewLikeOpInterface>]> {
  let summary = "Reserve tiles in a dataflow buffer for writing";
  let description = [{
    `ttl.cb_reserve` blocks until pages are available in the circular buffer
    for writing. Returns a tensor view of the reserved pages.

    This operation is used by producer threads (typically data movement kernels
    reading from DRAM) to acquire space in the CB before writing data.

    The result tensor has shape matching the CB's block shape and element type.
    The number of pages defaults to the CB's elements per block, but can be
    overridden via the optional `num_tiles` attribute for per-subblock reservation.

    Example:
    ```mlir
    %view = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sub = ttl.cb_reserve %cb {num_tiles = 3 : i64} : <[3, 3], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x3x!ttcore.tile<32x32, bf16>>
    ```
  }];
  let arguments = (ins
    TTL_CircularBuffer:$cb,
    OptionalAttr<I64Attr>:$num_tiles
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$cb attr-dict `:` type($cb) `->` type($result)";
  let hasVerifier = 1;
}

def TTL_CBPushOp : TTL_Op<"cb_push", [MemoryEffects<[MemWrite]>]> {
  let summary = "Signal that tiles have been written to the dataflow buffer";
  let description = [{
    `ttl.cb_push` signals that pages have been written to the circular buffer
    and are ready for consumer threads to read.

    This operation must be called after writing data to pages acquired via
    `ttl.cb_reserve`. It increments the CB's producer pointer.

    The number of pages defaults to the CB's elements per block, but can be
    overridden via the optional `num_tiles` attribute for per-subblock pushing.

    Example:
    ```mlir
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_push %cb {num_tiles = 3 : i64} : <[3, 3], !ttcore.tile<32x32, bf16>, 2>
    ```
  }];
  let arguments = (ins
    TTL_CircularBuffer:$cb,
    OptionalAttr<I64Attr>:$num_tiles
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
  let summary = "Store a tensor to a circular buffer reserve view";
  let description = [{
    `ttl.store` represents the user's intent to store a computed tensor into
    an output circular buffer.

    Emitted by Python's `o.store(result)` (overwrite) or `o += result`
    (accumulate). During `convert-ttl-to-compute`, this op is transformed
    into a `ttl.tile_store` inside the compute body.

    When `accumulate` is set, the enclosing loop is annotated for L1
    packer accumulation so that each iteration adds to the existing L1
    value instead of overwriting.

    Example:
    ```mlir
    %reserve = ttl.cb_reserve %cb : ... -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %result = ttl.add %a, %b : ...
    ttl.store %result, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                  tensor<1x1x!ttcore.tile<32x32, bf16>>
    // Accumulating store (from +=):
    ttl.store %result, %reserve {accumulate} : ...
    ```
  }];
  let arguments = (ins
    AnyRankedTensor:$tensor,
    AnyRankedTensor:$view,
    OptionalAttr<UnitAttr>:$accumulate
  );
  let assemblyFormat = "$tensor `,` $view attr-dict `:` type($tensor) `,` type($view)";
  let hasVerifier = 1;
}

def TTL_TileStoreOp : TTL_Op<"tile_store", [MemoryEffects<[MemWrite]>, TTL_DstResultOpTrait]> {
  let summary = "Store a tile into a circular buffer view";
  let description = [{
    `ttl.tile_store` packs a single tile from the DST register into a circular
    buffer view obtained from `ttl.cb_reserve`. This is the tile-level form of
    `ttl.store`, created by `convert-ttl-to-compute` from the tensor-level store.

    The tile value typically comes from compute operations (e.g., tile_add,
    tile_mul). The view must be from a prior `ttl.cb_reserve` call on the
    same CB.

    The `indices` operands are multi-dimensional CB tile coordinates derived
    from `ttl.iter_index` ops and the output indexing map. They are populated
    by `ttl-assign-dst` inside the `ttl.compute` body and resolved to concrete
    loop IVs by `ttl-lower-to-loops`. At `convert-ttl-to-ttkernel` time, the
    indices are linearized to a flat CB tile index for `pack_tile`.

    Example:
    ```mlir
    %view = ttl.cb_reserve %cb : <[2, 3], !ttcore.tile<32x32, bf16>, 1>
                                 -> tensor<2x3x!ttcore.tile<32x32, bf16>>
    // After assign-dst (inside ttl.compute body):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    ttl.tile_store %tile, %view[%i, %j] from dst[%c0]
      : !ttcore.tile<32x32, bf16>, tensor<2x3x!ttcore.tile<32x32, bf16>>
    // After lower-to-loops (iter_index replaced with loop IVs):
    ttl.tile_store %tile, %view[%row, %col] from dst[%c0]
      : !ttcore.tile<32x32, bf16>, tensor<2x3x!ttcore.tile<32x32, bf16>>
    ```
  }];
  let arguments = (ins
    AnyType:$tile,
    AnyRankedTensor:$view,
    Variadic<Index>:$indices,
    Index:$dst_index
  );
  let assemblyFormat = "$tile `,` $view `[` $indices `]` `from` `dst` `[` $dst_index `]` attr-dict `:` type($tile) `,` type($view)";
  let hasVerifier = 1;
}

#endif // TTLANG_DIALECT_TTL_IR_TTLOPS_TD
