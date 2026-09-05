// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPS_TD
#define TTLANG_DIALECT_TTL_IR_TTLOPS_TD

include "ttlang/Dialect/TTL/IR/TTLBase.td"
include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.td"
include "ttlang/Dialect/TTL/IR/TTLOpsTypes.td"
include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.td"
include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.td"
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
    - `dfb_id`, when present, identifies the logical dataflow buffer
      independently of the physical slot selected by `cb_index`. Bindings in
      different kernels with the same `dfb_id` refer to the same logical
      buffer.
    - `allocation_group`, when present, requests one compiler-verified physical
      allocation for all logical DFBs carrying the same group identity.
    - `tensor_backing`, when present, binds the complete dataflow buffer
      capacity to the specified byte range of an operation tensor argument.
      The range is relative to the local shard on each launch node. Tensor
      backing supports BF16, FP32, BFP4_B, and BFP8_B tile element types.
    Example:

    ```mlir
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    ```
  }];
  let arguments = (ins
    IndexAttr:$cb_index,
    DefaultValuedAttr<I64Attr, "2">:$block_count,
    OptionalAttr<IndexAttr>:$dfb_id,
    OptionalAttr<TTL_TensorBackingAttr>:$tensor_backing,
    OptionalAttr<TTL_DFBAllocationGroupAttr>:$allocation_group
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

    The result rank may be less than the tensor rank (a rank-reducing slice):
    the leading (tensor_rank - result_rank) dims are squeezed by scalar
    subscripts and the trailing dims map to the CB shape. Each squeezed scalar
    index feeds the full-rank tile-coordinate linearization, so a nonzero slot
    in a leading dim addresses the right sub-block. A range subscript in a
    squeezed position is rejected, since a range does not reduce rank.

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
  let summary = "Asynchronous copy involving a tensor, DFB, or pipe";
  let description = [{
    `ttl.copy` initiates an asynchronous transfer. Pipe-to-DFB receives return
    `!ttl.receive_request`; other copies return `!ttl.transfer_handle`.

    This operation is non-blocking. The destination is not safe to use until a
    corresponding `ttl.wait` has completed.

    A tensor transfer has exactly one DFB (`!ttl.cb`) operand and one ranked
    tensor operand with TTL layout encoding. A pipe send has a DFB source and a
    pipe destination. A pipe receive has a pipe source and a destination block
    reserved from a DFB. The copy transfers the complete payload in row-major
    tile order.

    The tensor shape normally equals one dataflow buffer block. A copy may span
    consecutive blocks by extending the tensor's innermost dimension by an
    integer multiple of the block's innermost dimension. All outer dimensions
    remain equal to the block shape. The span must not exceed the dataflow
    buffer's block count.

    An explicit positive `byte_count` copies exactly that many consecutive
    bytes starting at byte offset zero between two acquired dataflow-buffer
    blocks with different tile geometry. The source must be the exact block
    returned by `ttl.cb_wait`, and the destination must be the exact block
    returned by `ttl.cb_reserve`. Both dataflow buffers must use the same
    element data format, and the count must fit both acquired blocks. Later
    destination bytes remain unchanged.

    Pipe copies may also specify `byte_count` to transfer exactly that many
    consecutive bytes starting at byte offset zero of one source block.
    Byte-counted pipe transfers require a one-block transfer span. The sender
    and every receiver must specify the same count, and the count must fit both
    endpoint blocks and the unsigned 32-bit NoC transfer size. Omitting
    `byte_count` transfers the entire block.

    TODO(ttl): Add an optional TRID attribute (range 0..15) when TRID-specific
    TTKernel NOC ops are available. Issue: #87.

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
    AnyType:$dst,
    ConfinedAttr<OptionalAttr<I64Attr>,
                 [IntPositive, IntMaxValue<4294967295>]>:$byte_count
  );
  // The verifier selects the result type from the transfer direction.
  let results = (outs AnyType:$xf);
  let assemblyFormat = "$src `,` $dst attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TTL_WaitOp : TTL_Op<"wait", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Wait on a transfer handle";
  let description = [{
    `ttl.wait` blocks until the asynchronous transfer identified by the input
    transfer handle or receive request is complete and the destination is safe
    to use.

    Pipe receive requests use the compiler-managed PipeNet completion counter
    associated with the request. Other transfer handles use the barrier
    selected by their transfer direction.

    TODO(ttl): Add an optional TRID attribute (range 0..15) when TRID-specific
    barriers are available in TTKernel. Issue: #87.

    Example:

    ```mlir
    %xf = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ...
    ttl.wait %xf : !ttl.transfer_handle<read>
    ```

    Transfer direction is modeled in the transfer handle type
    (`!ttl.transfer_handle<read>` or `!ttl.transfer_handle<write>`). The
    verifier requires direction-typed handles so lowering can always emit
    exactly one barrier (read vs. write). Pipe receive requests use the
    direction-independent `!ttl.receive_request` type.
  }];
  // NOTE: Use AnyType here so assembly prints/parses the full type spelling
  // (`!ttl.transfer_handle<read/write>`) rather than the stripped custom form
  // (`<read>/<write>`) and does not require a custom printer. The verifier still
  // enforces the operand is a direction-typed transfer handle.
  let arguments = (ins AnyType:$xf);
  let assemblyFormat = "$xf attr-dict `:` type($xf)";
  let hasVerifier = 1;
}

def TTL_WaitAnyOp : TTL_Op<"wait_any",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Wait for one posted PipeNet receive";
  let description = [{
    Blocks until at least one request has completed. Candidates are inspected
    in cyclic operand order beginning at `start`, normalized modulo the number
    of requests. The first completed request in that order is selected. Only the
    selected request becomes complete; all other requests remain pending.

    The request list must be nonempty and contain distinct SSA values. Requests
    may originate from different PipeNets. Their logical receive identities and
    physical synchronization resources remain compiler-owned.
  }];
  let arguments = (ins Variadic<TTL_ReceiveRequest>:$requests, Index:$start);
  let results = (outs TTL_ReadyReceive:$ready);
  let assemblyFormat = [{
    $requests `start` $start attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def TTL_ReadyReceiveIndexOp : TTL_Op<"ready_receive_index", [Pure]> {
  let summary = "Return the selected receive-request index";
  let description = [{
    Returns the zero-based operand index selected by `ttl.wait_any`.
  }];
  let arguments = (ins TTL_ReadyReceive:$ready);
  let results = (outs Index:$index);
  let assemblyFormat = "$ready attr-dict `:` type($ready)";
}

//===----------------------------------------------------------------------===//
// Elementwise compute operations (shared ElementWise base)
//===----------------------------------------------------------------------===//

class TTL_ElementWiseBase<string mnemonic, string primitive,
                          list<Trait> traits>
    : TTL_ComputePrimitiveOp<mnemonic, primitive, traits> {
  let results = (outs AnyType:$result);
}

class TTL_ElementWiseBinary<string mnemonic,
                            string primitive = "ElementwiseBinary",
                            list<Trait> traits = []>
    : TTL_ElementWiseBase<
          mnemonic, primitive,
          !listconcat(traits, [Pure, SameOperandsAndResultType,
                               TTL_BinaryElementwiseOpTrait])> {
  let summary = "Binary elementwise operation";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

class TTL_ElementWiseUnary<string mnemonic, list<Trait> traits = []>
    : TTL_ElementWiseBase<
          mnemonic, "ElementwiseUnary",
          !listconcat(traits, [Pure, SameOperandsAndResultType,
                               TTL_UnaryElementwiseOpTrait])> {
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

    A body containing `ttl.tile_accumulate` requires at least one reduction
    iterator. The accumulator remains resident in DST across the corresponding
    reduction iterations.

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

    /// Returns the unique formal output index whose attached DFB matches
    /// `view`. Failure means the association is absent or ambiguous.
    /// TODO(#797): Encode this association in `ttl.tile_store` and remove
    /// DFB-based discovery.
    FailureOr<unsigned> getOutputIndexForView(Value view);
    FailureOr<unsigned> getOutputIndexForView(Value view, Operation *use);

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
    ParentOneOf<["ComputeOp", "DstSectionOp", "IfSrcOp", "IfDstOp",
                 "PipeNetForeachSrcOp", "PipeNetForeachDstOp",
                 "PipeNetScopeOp", "AccumulationScopeOp"]>]> {
  let summary = "Terminate a TTL region body";
  let description = [{
    `ttl.yield` terminates the body of `ttl.compute`, `ttl.dst_section`,
    `ttl.if_src`, `ttl.if_dst`, `ttl.pipenet_foreach_src`,
    `ttl.pipenet_foreach_dst`, `ttl.pipenet_scope`, and
    `ttl.accumulation_scope`. In `ttl.accumulation_scope`, it carries one
    updated value per accumulation output. Other TTL regions use an
    operand-less terminator.
  }];
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
  let hasVerifier = 1;
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}

//===----------------------------------------------------------------------===//
// Accumulation scope
//===----------------------------------------------------------------------===//

def TTL_AccumulationScopeOp : TTL_Op<"accumulation_scope", [
    AttrSizedOperandSegments,
    SingleBlockImplicitTerminator<"YieldOp">,
    RecursiveMemoryEffects,
    DeclareOpInterfaceMethods<TTL_AccumulationScopeOpInterface>
  ]> {
  let summary = "Structured accumulation scope";
  let description = [{
    `ttl.accumulation_scope` records an accumulation policy for one or more
    destination tensors. The operation does not prescribe whether the
    accumulated value is stored in DST, a dataflow buffer, or another
    target-specific storage mechanism.

    The `initial_modes` attribute contains one entry per output. `inits`
    contains one operand for each output whose initial mode is `init`, in output
    order.

    The body uses one block argument per output and yields one value per output.
    This form makes
    cross-output data dependence an SSA property: if the yielded value for one
    output uses another output's updated value, normal SSA use-def edges encode
    that dependency.
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$outputs,
    Variadic<AnyRankedTensor>:$inits,
    ArrayAttr:$initial_modes
  );
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DST register section
//===----------------------------------------------------------------------===//

def TTL_DstSectionOp : TTL_Op<"dst_section", [
    SingleBlockImplicitTerminator<"YieldOp">,
    RecursiveMemoryEffects,
    NoRegionArguments,
    DeclareOpInterfaceMethods<RegionBranchOpInterface>
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
  let summary = "Initialize SFPU unpacker/packer for CB data formats";
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
  let summary = "Acquire the DST register lock";
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
  let summary = "Commit the DST register lock (MATH thread)";
  let description = [{
    Release the DST register lock held by the MATH thread, mirroring
    `ttkernel.tile_regs_commit`. This is blocking and must be paired with a
    preceding `tile_regs_acquire`.
  }];
  let assemblyFormat = "attr-dict";
}

def TTL_TileRegsWaitOp : TTL_Op<"tile_regs_wait",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Wait for DST register availability (PACK thread)";
  let description = [{
    Acquire the DST register lock for the PACK thread, blocking until the MATH
    thread commits. Mirrors `ttkernel.tile_regs_wait`.
  }];
  let assemblyFormat = "attr-dict";
}

def TTL_TileRegsReleaseOp : TTL_Op<"tile_regs_release",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Release DST register lock (PACK thread)";
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
// Pipe operations (for multi-core communication)
//===----------------------------------------------------------------------===//

def TTL_IsDeviceOp : TTL_Op<"is_device", [Pure]> {
  let summary = "Test whether the kernel executes on a logical device";
  let description = [{
    Returns true when the current logical device matches `device` in `domain`.
    Logical device coordinates are runtime values supplied by the host. Target
    lowering determines how those coordinates map to physical devices.
  }];
  let arguments = (ins
    TTL_DeviceDomainAttr:$domain,
    TTL_DeviceRefAttr:$device
  );
  let results = (outs I1:$result);
  let assemblyFormat = "$device `in` $domain attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def TTL_CurrentDeviceIndexOp : TTL_Op<"current_device_index", [Pure]> {
  let summary = "Get the current logical device index";
  let description = [{
    Returns the zero-based row-major order of the current logical device in
    `domain`. Component and axis order follow the order stored in the domain.
    The result depends only on the logical device coordinates, not on physical
    device identifiers or fabric topology.
  }];
  let arguments = (ins TTL_DeviceDomainAttr:$domain);
  let results = (outs Index:$result);
  let assemblyFormat = "$domain attr-dict `:` type($result)";
}

def TTL_IsDeviceInRangeOp : TTL_Op<"is_device_in_range", [Pure]> {
  let summary = "Test whether the kernel executes in a logical device range";
  let description = [{
    Returns true when the current logical device is in the half-open `range`
    within `domain`. Logical device coordinates are runtime values supplied by
    the host. Target lowering determines how those coordinates map to physical
    devices.
  }];
  let arguments = (ins
    TTL_DeviceDomainAttr:$domain,
    TTL_DeviceRangeAttr:$range
  );
  let results = (outs I1:$result);
  let assemblyFormat = "$range `in` $domain attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def TTL_CreatePipeOp : TTL_Op<"create_pipe", [Pure]> {
  let summary = "Create a pipe for core-to-core communication";
  let description = [{
    Creates a pipe that defines a transfer relation from a source core to
    one or more receiver cores.

    A point-to-point pipe has one receiver. A collective pipe has a receiver
    set. The receiver-set syntax records the collective transfer contract; it
    does not determine which TTKernel NOC operation is emitted later.
    `isCollective = true` records a collective transfer contract when the
    receiver set contains a single point.

    Example:
    ```mlir
    // Point-to-point: (0,0) -> (1,0)
    %p = ttl.create_pipe src(0, 0) dst(1, 0) : !ttl.pipe

    // Collective: (0,0) -> receiver set (1,0)-(1,3)
    %p = ttl.create_pipe src(0, 0) dst(1, 0) to (1, 3) : !ttl.pipe
    ```
  }];
  let arguments = (ins
    I64Attr:$srcX,
    I64Attr:$srcY,
    I64Attr:$dstStartX,
    I64Attr:$dstStartY,
    I64Attr:$dstEndX,
    I64Attr:$dstEndY,
    I64Attr:$pipeNetId,
    // Optional user-facing name for the owning PipeNet, used only to
    // render diagnostics in source-level terms (e.g.
    // `a_pipe_net.is_active()` instead of `net_<id>.is_active()`).
    // No effect on op semantics.
    OptionalAttr<StrAttr>:$pipeNetName,
    OptionalAttr<BoolAttr>:$isCollective,
    OptionalAttr<TTL_DeviceTransferAttr>:$deviceTransfer
  );
  let results = (outs TTL_Pipe:$result);
  let assemblyFormat = [{
    `src` `(` $srcX `,` $srcY `)` `dst` `(` $dstStartX `,` $dstStartY `)` `to` `(` $dstEndX `,` $dstEndY `)` `net` $pipeNetId attr-dict `:` type($result)
  }];
  let hasVerifier = 1;

  let builders = [
    // Builder for a single-receiver pipe with pipeNetId = 0.
    OpBuilder<(ins "int64_t":$srcX, "int64_t":$srcY,
                   "int64_t":$dstX, "int64_t":$dstY), [{
      build($_builder, $_state, srcX, srcY, dstX, dstY, dstX, dstY, 0);
    }]>,
    // Builder with all coordinates (no name)
    OpBuilder<(ins "int64_t":$srcX, "int64_t":$srcY,
                   "int64_t":$dstStartX, "int64_t":$dstStartY,
                   "int64_t":$dstEndX, "int64_t":$dstEndY,
                   "int64_t":$pipeNetId), [{
      auto pipeType = PipeType::get($_builder.getContext(),
                                    srcX, srcY, dstStartX, dstStartY,
                                    dstEndX, dstEndY, pipeNetId);
      build($_builder, $_state, pipeType,
            $_builder.getI64IntegerAttr(srcX),
            $_builder.getI64IntegerAttr(srcY),
            $_builder.getI64IntegerAttr(dstStartX),
            $_builder.getI64IntegerAttr(dstStartY),
            $_builder.getI64IntegerAttr(dstEndX),
            $_builder.getI64IntegerAttr(dstEndY),
            $_builder.getI64IntegerAttr(pipeNetId),
            /*pipeNetName=*/StringAttr(),
            /*isCollective=*/BoolAttr(),
            /*deviceTransfer=*/DeviceTransferAttr());
    }]>
  ];
}

def TTL_IfSrcOp : TTL_Op<"if_src",
    [SingleBlockImplicitTerminator<"YieldOp">,
     DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "Execute region if current core is the source of a pipe";
  let description = [{
    Executes the body region if the current core's coordinates match the
    source coordinates of the given pipe. Used for implementing pipe-based
    communication patterns where the source core sends data.

    The body region has no block arguments. Inside the body, the pipe's
    destination coordinates can be accessed from the pipe type directly.

    Example:
    ```mlir
    %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) : !ttl.pipe<...>
    ttl.if_src %p : !ttl.pipe<...> {
      // Only executes on core (0, 0)
      // Can use ttl.copy to send data through the pipe
    }
    ```
  }];
  let arguments = (ins TTL_Pipe:$pipe);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $pipe `:` type($pipe) $body attr-dict
  }];
}

def TTL_IfDstOp : TTL_Op<"if_dst",
    [SingleBlockImplicitTerminator<"YieldOp">,
     DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "Execute region if current core is a destination of a pipe";
  let description = [{
    Executes the body region if the current core's coordinates fall within
    the destination range of the given pipe. Used for implementing pipe-based
    communication patterns where destination cores receive data.

    The body region has no block arguments. Inside the body, the pipe's
    source coordinates can be accessed from the pipe type directly.

    Example:
    ```mlir
    %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) : !ttl.pipe<...>
    ttl.if_dst %p : !ttl.pipe<...> {
      // Executes on cores (1, 0), (1, 1), (1, 2), (1, 3)
      // Can use ttl.copy to receive data from the pipe
    }
    ```
  }];
  let arguments = (ins TTL_Pipe:$pipe);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $pipe `:` type($pipe) $body attr-dict
  }];
}

def TTL_PipeNetForeachSrcOp : TTL_Op<"pipenet_foreach_src",
    [SingleBlockImplicitTerminator<"YieldOp">,
     DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "Iterate over source pipe records in a PipeNet";
  let description = [{
    Executes the body region once for each static PipeNet record whose source
    coordinate matches the current launch node. A graph record additionally
    requires the current logical device to match its source endpoint. The body
    receives a `!ttl.selected_pipe_src` argument representing the selected
    record for that iteration.

    The records in `$records` are ordered by user PipeNet construction
    order. That order defines program order for a launch node that matches
    multiple records.
  }];

  let arguments = (ins
    TTL_PipeNetRecordsAttr:$records
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def TTL_PipeNetForeachDstOp : TTL_Op<"pipenet_foreach_dst",
    [SingleBlockImplicitTerminator<"YieldOp">,
     DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "Iterate over destination pipe records in a PipeNet";
  let description = [{
    Executes the body region once for each static PipeNet record whose
    destination range contains the current launch node. A graph record
    additionally requires the current logical device to match its destination
    endpoint. The body receives a `!ttl.selected_pipe_dst` argument
    representing the selected record for that iteration.

    The records in `$records` are ordered by user PipeNet construction
    order. That order defines program order for a launch node that matches
    multiple records.
  }];

  let arguments = (ins
    TTL_PipeNetRecordsAttr:$records
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

class TTL_SelectPipeOp<string mnemonic, string role, Type pipeType>
    : TTL_Op<mnemonic, [Pure]> {
  let summary = "Represent a record selected by " # role # " iteration";
  let description = "Represents a PipeNet record selected for " # role #
      " iteration. The operands contain the selected record fields. "
      "`$recordIndex` identifies the record in `$records`; the remaining "
      "operands contain its coordinates and receiver count. "
      "`$srcInDstRange` is true when the source coordinate belongs to the "
      "destination range. `$records` identifies the PipeNet and transfer "
      "contract. The device-index operands identify the logical-device edge "
      "associated with the selected record. They remain explicit operands for "
      "all records so selected-pipe IR has one stable operand contract; "
      "device-index accessors separately require graph records.";
  let arguments = (ins
    Index:$recordIndex,
    Index:$srcX,
    Index:$srcY,
    Index:$dstStartX,
    Index:$dstStartY,
    Index:$dstEndX,
    Index:$dstEndY,
    Index:$numDests,
    I1:$srcInDstRange,
    Index:$sourceDeviceIndex,
    Index:$destinationDeviceIndex,
    TTL_PipeNetRecordsAttr:$records
  );
  let results = (outs pipeType:$pipe);
  let assemblyFormat = [{
    `record` `(` $recordIndex `)` `src` `(` $srcX `,` $srcY `)` `dst` `(` $dstStartX `,` $dstStartY `)` `to` `(` $dstEndX `,` $dstEndY `)`
    `num_dests` `(` $numDests `)` `src_in_dst` `(` $srcInDstRange `)`
    `devices` `(` $sourceDeviceIndex `,` $destinationDeviceIndex `)` attr-dict `:` type($pipe)
  }];
}

def TTL_SelectPipeSrcOp
    : TTL_SelectPipeOp<"select_pipe_src", "source", TTL_SelectedPipeSrc>;

def TTL_SelectPipeDstOp
    : TTL_SelectPipeOp<"select_pipe_dst", "destination", TTL_SelectedPipeDst>;

def TTL_SelectedPipeSourceDeviceIndexOp
    : TTL_Op<"selected_pipe_source_device_index", [Pure]> {
  let summary = "Get the selected pipe record's logical source device index";
  let description = [{
    Returns the row-major logical-device index of the source endpoint stored in
    the selected graph PipeNet record.
  }];
  let arguments = (ins TTL_SelectedPipeDst:$pipe);
  let results = (outs Index:$result);
  let assemblyFormat = "$pipe attr-dict `:` type($pipe)";
  let hasVerifier = 1;
}

def TTL_SelectedPipeDestinationDeviceIndexOp
    : TTL_Op<"selected_pipe_destination_device_index", [Pure]> {
  let summary = "Get the selected pipe record's logical destination device index";
  let description = [{
    Returns the row-major logical-device index of the destination endpoint
    stored in the selected graph PipeNet record.
  }];
  let arguments = (ins TTL_SelectedPipeSrc:$pipe);
  let results = (outs Index:$result);
  let assemblyFormat = "$pipe attr-dict `:` type($pipe)";
  let hasVerifier = 1;
}

def TTL_SelectedPipeSourceCoordinatesOp
    : TTL_Op<"selected_pipe_source_coordinates", [Pure]> {
  let summary = "Get the selected pipe record's source node coordinates";
  let description = [{
    Returns the source node coordinates stored in the selected PipeNet record.
  }];
  let arguments = (ins TTL_SelectedPipeDst:$pipe);
  let results = (outs Index:$x, Index:$y);
  let assemblyFormat = "$pipe attr-dict `:` type($pipe)";
  let hasVerifier = 1;
}

def TTL_SelectedPipeDestinationCoordinatesOp
    : TTL_Op<"selected_pipe_destination_coordinates", [Pure]> {
  let summary = "Get the selected pipe record's destination node coordinates";
  let description = [{
    Returns the inclusive start and end coordinates of the destination node
    range stored in the selected PipeNet record. Point destinations have equal
    start and end coordinates.
  }];
  let arguments = (ins TTL_SelectedPipeSrc:$pipe);
  let results = (outs Index:$startX, Index:$startY, Index:$endX, Index:$endY);
  let assemblyFormat = "$pipe attr-dict `:` type($pipe)";
  let hasVerifier = 1;
}

def TTL_PipeTransferCreateOp : TTL_Op<"pipe_transfer.create", [Pure]> {
  let summary = "Create a pipe transfer reference";
  let description = [{
    Creates a reference used by a pipe send or receiver post. A complete
    transfer consists of one send and one corresponding post for each receiver.
    The sender and receiver callbacks may use different references because each
    constructs its pipe value independently. The attributes record the common
    point-to-point or collective contract and the optional logical-device
    transfer. The logical-device transfer is absent for transfers within one
    device.

    `block_span` is the number of consecutive original DFB blocks delivered by
    one transfer. Scalar transfers use the default span of one. Grouping
    transformations increase the span while preserving the original DFB block
    geometry.

    `destination_group_depth` is the number of complete transfers that the
    transport schedule may leave resident in each receiver DFB. The default
    depth of one preserves lockstep transfer semantics. A greater depth permits
    multiple grouped transfers to occupy distinct receiver slots.

    Record-selected pipe references require the default span and destination
    depth because one operation selects one scalar record at runtime. Grouped
    transfers require a statically identified pipe reference.
  }];
  let arguments = (ins
    TTL_PipeReference:$pipe,
    TTL_PipeTransferKindAttr:$kind,
    OptionalAttr<TTL_DeviceTransferAttr>:$deviceTransfer,
    DefaultValuedAttr<ConfinedAttr<I64Attr, [IntPositive]>, "1">:$block_span,
    DefaultValuedAttr<ConfinedAttr<I64Attr, [IntPositive]>, "1">:
         $destination_group_depth
  );
  let results = (outs TTL_PipeTransfer:$transfer);
  let assemblyFormat = "$pipe attr-dict `:` type($pipe) `->` type($transfer)";
  let hasVerifier = 1;
}

def TTL_PipeTransferPostOp : TTL_Op<"pipe_transfer.post",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Post a pipe receive destination";
  let description = [{
    Registers the receiver-owned destination DFB slot for a pipe transfer and
    returns the token that represents receiver-side completion. The selected
    address and synchronization protocols determine whether lowering publishes
    the address or signals sender readiness. When `byte_count` is present, the
    matching sender writes exactly that many consecutive bytes starting at byte
    offset zero of the reserved destination block. Later bytes remain
    unchanged.
  }];
  let arguments = (ins
    TTL_PipeTransfer:$transfer,
    AnyType:$dst,
    ConfinedAttr<OptionalAttr<I64Attr>,
                 [IntPositive, IntMaxValue<4294967295>]>:$byte_count
  );
  let results = (outs TTL_PipeToken:$token);
  let assemblyFormat = "$transfer `,` $dst attr-dict `:` functional-type(operands, results)";
}

def TTL_PipeTransferSendOp : TTL_Op<"pipe_transfer.send",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Send a pipe transfer payload";
  let description = [{
    Waits until the selected synchronization condition is satisfied, then
    writes the source DFB payload to the receiver-owned destination storage and
    signals receiver completion. The returned write handle preserves the
    existing `ttl.wait` ordering contract for send-side code. When `byte_count`
    is present, the send reads exactly that many consecutive bytes starting at
    byte offset zero of each source block.
  }];
  let arguments = (ins
    TTL_PipeTransfer:$transfer,
    TTL_CircularBuffer:$src,
    ConfinedAttr<OptionalAttr<I64Attr>,
                 [IntPositive, IntMaxValue<4294967295>]>:$byte_count
  );
  let results = (outs TTL_TransferHandle:$xf);
  let assemblyFormat = "$transfer `,` $src attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TTL_PipeTransferWaitOp : TTL_Op<"pipe_transfer.wait",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Wait for pipe receive completion";
  let description = [{
    Blocks until the posted receive represented by the pipe token has completed.
    The receiver-owned destination DFB slot is safe to consume after this
    operation completes.
  }];
  let arguments = (ins
    TTL_PipeToken:$token
  );
  let assemblyFormat = "$token attr-dict `:` type($token)";
}

def TTL_PipeTransferWaitAnyOp : TTL_Op<"pipe_transfer.wait_any",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Wait for one pipe receive token";
  let description = [{
    Blocks until one candidate completion counter has reached the sequence in
    its token. Candidates are inspected in cyclic operand order beginning at
    `start`. Tokens may name different PipeNets. The result identifies the
    selected candidate index and does not complete nonselected tokens.
  }];
  let arguments = (ins Variadic<TTL_PipeToken>:$tokens, Index:$start);
  let results = (outs TTL_ReadyReceive:$ready);
  let assemblyFormat = [{
    $tokens `start` $start attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def TTL_PipeNetScopeOp : TTL_Op<"pipenet_scope",
    [SingleBlockImplicitTerminator<"YieldOp">,
     DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "Mark a DFB region restricted to PipeNet endpoints";
  let description = [{
    Marks a region whose dataflow-buffer acquire/release sequence must execute
    only on specified source or destination nodes of a PipeNet. The operation
    does not change runtime behavior; it records these execution requirements
    for the enclosed region.

    The requirements are stored in parallel arrays:

    - `ttl.pipe_net_ids`: array of PipeNet ids.
    - `ttl.pipe_net_roles`: array of endpoint selectors, where 0 is source and
      1 is destination.
  }];
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def TTL_IsSrcOp : TTL_Op<"is_src",
    [Pure, DeclareOpInterfaceMethods<TTL_PipeNetPredicateOpInterface>]> {
  let summary = "Predicate: current node is a source of any pipe in the PipeNet";
  let description = [{
    Returns `true` on nodes that match the `src` of at least one pipe in the
    named PipeNet, `false` elsewhere. The value-returning sibling of
    `ttl.if_src`; used as the condition of an `scf.if` to guard pipe-coupled
    work without entering a region. When `records` is present, the current
    logical device must also match the source endpoint in a record.
  }];
  let arguments = (ins I64Attr:$pipe_net_id,
                        OptionalAttr<TTL_PipeNetRecordsAttr>:$records);
  let results = (outs I1:$result);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def TTL_IsDstOp : TTL_Op<"is_dst",
    [Pure, DeclareOpInterfaceMethods<TTL_PipeNetPredicateOpInterface>]> {
  let summary = "Predicate: current node is a destination of any pipe in the PipeNet";
  let description = [{
    Returns `true` on nodes whose coordinate falls within the `dst` range of
    at least one pipe in the named PipeNet, `false` elsewhere. The
    value-returning sibling of `ttl.if_dst`. When `records` is present, the
    current logical device must also match the destination endpoint in a
    record.
  }];
  let arguments = (ins I64Attr:$pipe_net_id,
                        OptionalAttr<TTL_PipeNetRecordsAttr>:$records);
  let results = (outs I1:$result);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def TTL_IsActiveOp : TTL_Op<"is_active",
    [Pure, DeclareOpInterfaceMethods<TTL_PipeNetPredicateOpInterface>]> {
  let summary = "Predicate: current node is a source or destination in the PipeNet";
  let description = [{
    Returns `true` on nodes that are either a source or a destination of at
    least one pipe in the named PipeNet (the union of `is_src` and `is_dst`).
    When `records` is present, node roles also include the corresponding
    logical-device endpoints.
  }];
  let arguments = (ins I64Attr:$pipe_net_id,
                        OptionalAttr<TTL_PipeNetRecordsAttr>:$records);
  let results = (outs I1:$result);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
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

def TTL_DPrintOp
    : TTL_Op<"dprint", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Debug print from device kernel";
  let description = [{
    Emits device debug print calls. Mode determines the output format:
    - "scalar": prints format string with scalar values
    - "cb": prints CB metadata (size, limit, ptrs)
    - "tile": prints physical tile contents from a DFB using the operand dimensions
    - "tensor": prints raw memory pages from tensor backing buffer
    - "dst": dumps live DST register slots

    Example:
    ```mlir
    ttl.dprint "x = {}" (%x) {mode = "scalar"} : (index)
    ttl.dprint "input" (%cb) {mode = "cb"} : (!ttl.cb<...>)
    ttl.dprint "after exp" () {mode = "dst"}
    ```

    Debug output is an observable side effect. Transformations must preserve
    its order relative to tensor evaluation unless they move the print with
    the observed computation.
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
class TTL_TileOp<string mnemonic, string primitive,
                 list<Trait> traits = []>
    : TTL_ComputePrimitiveOp<
          mnemonic, primitive,
          !listconcat(traits, [TTL_TileComputeOpTrait,
                               TTL_DstAccessOpInterface,
                               TTL_TileExecutionOpInterface])>;

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
                             TTL_DstResultOpTrait,
                             TTL_DstAccessOpInterface,
                             TTL_TileExecutionOpInterface]> {
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
                            TTL_DataMovementOpTrait,
                            TTL_DstAccessOpInterface,
                            TTL_TileExecutionOpInterface]> {
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

def TTL_DstIndexOp : TTL_Op<"dst_index",
    [Pure, TTL_DstAccessOpInterface, TTL_TileExecutionOpInterface,
     AllTypesMatch<["source", "result"]>]> {
  let summary = "Index a tile value resident in a specific DST slot";
  let description = [{
    `ttl.dst_index` names an existing DST-resident tile at `dst_index` while
    preserving SSA dependence on `source`. It has no hardware side effect and
    does not change the tile value.
  }];
  let arguments = (ins TTL_TileType:$source, Index:$dst_index);
  let results = (outs TTL_TileType:$result);
  let assemblyFormat =
      "$source `[` $dst_index `]` attr-dict `:` "
      "type($source) `->` type($result)";
}

// Common base for tile binary ops. Carries traits that hold unconditionally
// (binary, writes to DST, all types match). Subclasses add the input-source
// trait that reflects the op's actual hardware behavior.
class TTL_TileBinaryOpBase<string mnemonic,
                           string primitive = "ElementwiseBinary",
                           list<Trait> traits = []>
    : TTL_TileOp<mnemonic, primitive,
                 !listconcat(traits, [TTL_TileBinaryOpTrait,
                                      TTL_DstResultOpTrait,
                                      AllTypesMatch<["lhs", "rhs", "result"]>])> {
  let summary = "Binary tile operation";
  let arguments = (ins TTL_TileType:$lhs, TTL_TileType:$rhs, Index:$dst_index);
  let results = (outs TTL_TileType:$result);
  let assemblyFormat = "$lhs `,` $rhs `into` `dst` `[` $dst_index `]` attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

// Always-SFPU binary tile op: reads both operands from DST unconditionally.
// Use for ops with no FPU equivalent (div, max, min).
class TTL_TileSfpuBinaryOp<string mnemonic, list<Trait> traits = []>
    : TTL_TileBinaryOpBase<mnemonic, "ElementwiseBinary",
                           !listconcat(traits, [TTL_DSTInputsTrait])>;

// The selected FPU/SFPU strategy determines whether operands are consumed from
// dataflow buffers or DST.
class TTL_TileStrategyDepBinaryOp<string mnemonic, string primitive,
                                  list<Trait> traits = []>
    : TTL_TileBinaryOpBase<mnemonic, primitive,
                           !listconcat(traits,
                                       [TTL_StrategyDependentBinaryOpTrait])>;

class TTL_TileUnaryOp<string mnemonic, list<Trait> traits = []>
    : TTL_TileOp<mnemonic, "ElementwiseUnary",
                 !listconcat(traits, [TTL_TileUnaryOpTrait,
                                      TTL_DSTInputsTrait, TTL_InPlaceOpTrait,
                                      TTL_DstResultOpTrait,
                                      AllTypesMatch<["input", "result"]>])> {
  let summary = "Unary tile operation";
  let arguments = (ins TTL_TileType:$input, Index:$dst_index);
  let results = (outs TTL_TileType:$result);
  let assemblyFormat = "$input `into` `dst` `[` $dst_index `]` attr-dict `:` type($input) `->` type($result)";
}

def TTL_TileAccumulateOp : TTL_TileOp<"tile_accumulate", "ElementwiseBinary",
    [TTL_TileBinaryOpTrait, TTL_InPlaceOpTrait,
     TTL_AccumulatingOpTrait, TTL_DstResultOpTrait,
     AllTypesMatch<["accumulator", "contribution", "result"]>]> {
  let summary = "In-DST tile recurrence";
  let description = [{
    `ttl.tile_accumulate` models one tile-level recurrence update whose
    accumulator is resident in DST. The operation applies `combiner` to
    `accumulator` and `contribution`, writes the result into `dst_index`, and
    returns the value resident in that same DST slot. Supported combiners are
    `add` and `max`.

    Example:

    ```mlir
    %idx = arith.constant 0 : index
    %next = ttl.tile_accumulate %acc, %delta add into dst[%idx]
        : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>
          -> !ttcore.tile<32x32, bf16>
    ```
  }];
  let arguments = (ins TTL_TileType:$accumulator,
                       TTL_TileType:$contribution,
                       TTL_AccumulationCombinerAttr:$combiner,
                       Index:$dst_index);
  let results = (outs TTL_TileType:$result);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Binary Elementwise Operations (Tensor + Tile Ops)
//===----------------------------------------------------------------------===//

// Multiclass for tensor operations paired with FPU/SFPU tile operations.
// Add, subtract, and multiply select one tile execution strategy before DST
// assignment.
multiclass TTL_BinaryStrategyDepElementwisePair<string mnemonic,
                                                string kernelOp,
                                                string primitive,
                                                list<Trait> opTraits = []> {
  def Op : TTL_ElementWiseBinary<mnemonic, primitive, opTraits> {
    let description = "Elementwise "#mnemonic#" on tensors. Lowered to ttl.compute with ttl.tile_"#mnemonic#".";
  }
  def TileOp : TTL_TileStrategyDepBinaryOp<"tile_"#mnemonic, primitive> {
    let description = "Tile-level "#mnemonic#". Maps to ttkernel."#kernelOp#".";
  }
}

// Multiclass for paired tensor + always-SFPU tile binary ops.
// Used by div/max/min, which have no FPU lowering today.
multiclass TTL_BinarySfpuElementwisePair<string mnemonic, string kernelOp,
                                         list<Trait> extraTraits = []> {
  def Op : TTL_ElementWiseBinary<mnemonic> {
    let description = "Elementwise "#mnemonic#" on tensors. Lowered to ttl.compute with ttl.tile_"#mnemonic#".";
  }
  def TileOp : TTL_TileSfpuBinaryOp<"tile_"#mnemonic, extraTraits> {
    let description = "Tile-level "#mnemonic#". Maps to ttkernel."#kernelOp#".";
  }
}

// Define all binary elementwise ops using the multiclasses
defm TTL_Add : TTL_BinaryStrategyDepElementwisePair<"add", "add_tiles", "Add">;
defm TTL_Sub : TTL_BinaryStrategyDepElementwisePair<"sub", "sub_tiles", "Subtract">;
defm TTL_Mul : TTL_BinaryStrategyDepElementwisePair<"mul", "mul_tiles", "Multiply",
    [DeclareOpInterfaceMethods<TTL_DFBInputOpInterface>]>;
defm TTL_Div : TTL_BinarySfpuElementwisePair<"div", "div_binary_tile">;
defm TTL_Max : TTL_BinarySfpuElementwisePair<"max", "max_tiles",
                                             [TTL_InPlaceOpTrait]>;
defm TTL_Min : TTL_BinarySfpuElementwisePair<"min", "binary_min_tile",
                                             [TTL_InPlaceOpTrait]>;

// Elementwise compare: maps to TTKernel's native SFPU comparison ops.
defm TTL_Eq : TTL_BinarySfpuElementwisePair<"eq", "eq_binary_tile">;
defm TTL_Ne : TTL_BinarySfpuElementwisePair<"ne", "ne_binary_tile">;
defm TTL_Gt : TTL_BinarySfpuElementwisePair<"gt", "gt_binary_tile">;
defm TTL_Lt : TTL_BinarySfpuElementwisePair<"lt", "lt_binary_tile">;


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

defm TTL_Exp2 : TTL_UnaryElementwisePair<"exp2", "exp2_tile">;
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
defm TTL_Ceil : TTL_UnaryElementwisePair<"ceil", "ceil_tile">;
defm TTL_Sign : TTL_UnaryElementwisePair<"sign", "sign_tile">;
defm TTL_Gelu : TTL_UnaryElementwisePair<"gelu", "gelu_tile">;
defm TTL_Silu : TTL_UnaryElementwisePair<"silu", "silu_tile">;
defm TTL_Hardsigmoid : TTL_UnaryElementwisePair<"hardsigmoid", "hardsigmoid_tile">;
defm TTL_Expm1 : TTL_UnaryElementwisePair<"expm1", "expm1_tile">;
defm TTL_Square : TTL_UnaryElementwisePair<"square", "square_tile">;
defm TTL_Softsign : TTL_UnaryElementwisePair<"softsign", "softsign_tile">;
defm TTL_Signbit : TTL_UnaryElementwisePair<"signbit", "signbit_tile">;
defm TTL_Frac : TTL_UnaryElementwisePair<"frac", "frac_tile">;
defm TTL_Trunc : TTL_UnaryElementwisePair<"trunc", "trunc_tile">;

//===----------------------------------------------------------------------===//
// Exp Operations (Tensor + Tile)
//===----------------------------------------------------------------------===//
//
// `exp` is broken out of the generic unary multiclass because it carries
// hardware SFPU flags that the generic ops do not. The flags are forwarded
// unchanged from the tensor op to the tile op and then to the TTKernel
// exp_tile / exp_tile_init ops during lowering. All flags are optional or
// default-valued so that `ttl.exp %x` (and tile form) keep their plain
// spelling when no flags are set.
//
// Flags (mirroring tt-metal ckernel::exp_tile / exp_tile_init):
//   approx          - enable the fast approximate exp.
//   scale           - optional scale factor `s`; when present the op computes
//                     exp(s * x) (sets scale_en at the TTKernel level).
//   input_clamping  - clamp behaviour for very negative inputs (only relevant
//                     for approximate mode); defaults to ClampToNegative.
//   iterations      - number of SFPU lane iterations (default 8).

// Shared flag arguments for both the tensor and tile exp ops.
defvar TTL_ExpFlags = (ins
    DefaultValuedAttr<BoolAttr, "false">:$approx,
    OptionalAttr<F32Attr>:$scale,
    DefaultValuedAttr<TTL_InputClampingAttr,
        "::mlir::tt::ttl::InputClamping::ClampToNegative">:$input_clamping,
    DefaultValuedAttr<I32Attr, "8">:$iterations);

def TTL_ExpOp : TTL_ElementWiseUnary<"exp"> {
  let description = [{
    Elementwise exp on tensors. Lowered to ttl.compute with ttl.tile_exp.
    Carries optional hardware exp flags (approx / scale / input_clamping /
    iterations) that are forwarded to the tile op during lowering.
  }];
  let arguments = !con((ins AnyType:$input), TTL_ExpFlags);
}

def TTL_ExpTileOp : TTL_TileUnaryOp<"tile_exp"> {
  let description = [{
    Tile-level exp. Maps to ttkernel.exp_tile. Carries the same hardware exp
    flags as ttl.exp; they are forwarded to the TTKernel exp_tile / exp_tile_init
    ops during conversion.
  }];
  let arguments = !con((ins TTL_TileType:$input, Index:$dst_index),
                       TTL_ExpFlags);
}

//===----------------------------------------------------------------------===//
// Typecast Operations (Tensor + Tile)
//===----------------------------------------------------------------------===//
//
// `ttl.typecast` is a unary elementwise op that converts the data type of
// every element. The tile-level form `ttl.tile_typecast` carries the
// destination dtype implicitly via its result tile element type. Because
// input and result element types can differ, typecast does not use the standard
// unary elementwise multiclass (which constrains all operand and result
// element types to match).

def TTL_TypecastOp : TTL_ComputePrimitiveOp<"typecast", "Typecast",
    [Pure, TTL_UnaryElementwiseOpTrait]> {
  let summary = "Elementwise typecast on tensors";
  let description = [{
    Convert each element of `input` to the result element type. The result
    tensor has the same shape as the input. Identity typecasts fold to the
    input.
    Lowered to `ttl.compute` containing a `ttl.tile_typecast` operation.
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$input attr-dict `:` functional-type(operands, results)";
  let hasFolder = 1;
}

def TTL_TileTypecastOp : TTL_TileOp<"tile_typecast", "Typecast",
    [TTL_TileUnaryOpTrait, TTL_DSTInputsTrait, TTL_InPlaceOpTrait,
     TTL_DstResultOpTrait]> {
  let summary = "Tile-level typecast operation";
  let description = [{
    Convert the data type of a tile in DST. Maps to ttkernel.typecast_tile.
    Operates in-place on the DST register at `dst_index`; the destination
    dtype is encoded in the result tile element type.
  }];
  let arguments = (ins TTL_TileType:$input, Index:$dst_index);
  let results = (outs TTL_TileType:$result);
  let assemblyFormat =
      "$input `into` `dst` `[` $dst_index `]` attr-dict `:` "
      "type($input) `->` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Broadcast Operation
//===----------------------------------------------------------------------===//

def TTL_BlockBroadcastOp : TTL_ComputePrimitiveOp<"block.broadcast", "Broadcast", [Pure,
    DeclareOpInterfaceMethods<TTL_DFBInputOpInterface>]> {
  let summary = "Broadcast a block over specified dimensions to a target shape";
  let description = [{
    Broadcasts `$input` over each dimension listed in `$dims` to produce a
    result of shape `$shape`. For every dimension `d` in `$dims` the
    corresponding `input.shape[d]` must equal 1; for every dimension not in
    `$dims` the input dimension must equal `shape[d]`.

    For tiled blocks broadcast happens in two steps: intra-tile scalar
    broadcast for any innermost dimension listed in `$dims`, and inter-tile
    broadcast for every other dimension where the target shape is greater
    than 1. Row-major blocks are not supported.

    Example:
    ```mlir
    %r = ttl.block.broadcast %a dims = [1], shape = [4, 8]
           : tensor<4x1x!ttcore.tile<32x32, f32>>
          -> tensor<4x8x!ttcore.tile<32x32, f32>>
    ```
  }];
  let arguments = (ins AnyRankedTensor:$input,
                       DenseI64ArrayAttr:$dims,
                       DenseI64ArrayAttr:$shape);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input `dims` `=` $dims `,` `shape` `=` $shape attr-dict `:`
    type($input) `->` type($result)
  }];
  let hasVerifier = 1;
}

def TTL_TileBcastOp : TTL_TileOp<"tile_bcast", "Broadcast",
    [TTL_CBInputTileOpTrait, TTL_DstResultOpTrait]> {
  let summary = "Tile-level broadcast operation";
  let description = [{
    Broadcasts a row, column, or scalar within the output tile dimensions.
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

def TTL_ReduceOp : TTL_ComputePrimitiveOp<"reduce", "Reduce", [Pure,
    DeclareOpInterfaceMethods<TTL_DFBInputOpInterface>]> {
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
    Leading dimensions use inter-tile accumulation. Reducing either of the
    innermost dimensions also applies the corresponding within-tile hardware
    reduction.

    The input must be a CB-attached tensor of rank 2 or greater. The scaler is
    a CB-attached rank-2 single-tile tensor used by the hardware reduction.
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

def TTL_TileReduceOp : TTL_TileOp<"tile_reduce", "Reduce",
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

def TTL_TileRowNormalizationBlockOp :
    TTL_TileOp<"tile_row_normalization_block",
               "Reduce",
               [TTL_CBInputTileOpTrait, TTL_DstResultOpTrait]> {
  let summary = "Normalize one static row in one DST acquisition";
  let description = [{
    Computes `input * rsqrt(sum(input * input) * scale + epsilon)` for one
    rank-2 row of tiles. The optional gamma tensor is multiplied into the
    normalized result before publication.

    `dst_index` identifies the first output slot. The operation initially
    stores the inverse RMS scalar there, moves it to a compute source register,
    clears the acquired DST section, and writes the normalized row to the
    contiguous slots beginning at `dst_index`. `num_tiles` records the exact
    row width because scalarization replaces tensor operands with tile values.
    The input, gamma, and output operands identify dataflow buffers; gamma is
    equal to input when `has_gamma` is false and is not read in that mode.

    The operation is internal to block compute lowering. Its result represents
    the complete contiguous DST footprint, not one independently executable
    tile iteration.
  }];
  let arguments = (ins TTL_TileOrTensor:$input,
                       TTL_TileOrTensor:$gamma,
                       TTL_TileOrTensor:$output,
                       F32Attr:$scale,
                       F32Attr:$epsilon,
                       BoolAttr:$has_gamma,
                       ConfinedAttr<I64Attr, [IntPositive]>:$num_tiles,
                       Index:$dst_index);
  let results = (outs TTL_TileOrTensor:$result);
  let assemblyFormat = [{
    $input `,` $gamma `,` $output
    `scale` `=` $scale `epsilon` `=` $epsilon
    `has_gamma` `=` $has_gamma
    `num_tiles` `=` $num_tiles
    `into` `dst` `[` $dst_index `]` attr-dict `:`
    type($input) `,` type($gamma) `,` type($output) `->` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Transpose Operation
//===----------------------------------------------------------------------===//

def TTL_TransposeOp : TTL_ComputePrimitiveOp<"transpose", "Transpose", [Pure,
    DeclareOpInterfaceMethods<TTL_DFBInputOpInterface>]> {
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

def TTL_TileTransposeOp : TTL_TileOp<"tile_transpose", "Transpose",
    [TTL_CBInputTileOpTrait, TTL_DstResultOpTrait]> {
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

def TTL_FillOp : TTL_ComputePrimitiveOp<"fill", "Fill", [Pure]> {
  let summary = "Fill a tensor with a constant value";
  let description = [{
    Fills all elements of a tensor with the given f32 value.
    The output CB is determined by the ttl.store on this op's result.
  }];
  let arguments = (ins F32Attr:$value);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def TTL_TileFillOp : TTL_TileOp<"tile_fill", "Fill",
    [TTL_DstResultOpTrait]> {
  let summary = "Tile-level fill operation";
  let description = [{
    Fills a DST register tile with a constant f32 value.
    Maps to ttkernel.fill_tile_init + ttkernel.fill_tile.
  }];
  let arguments = (ins F32Attr:$value, Index:$dst_index);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$value `into` `dst` `[` $dst_index `]` attr-dict `:` type($result)";
}

def TTL_TileMulUnaryConstOp : TTL_TileOp<"tile_mul_unary_const",
    "MultiplyByConstant",
    [TTL_TileUnaryOpTrait, TTL_DSTInputsTrait, TTL_InPlaceOpTrait,
     TTL_DstResultOpTrait, AllTypesMatch<["input", "result"]>]> {
  let summary = "Tile-level multiply by a compile-time scalar constant";
  let description = [{
    Multiplies every element of a DST register tile by a compile-time
    scalar constant. Operates in-place at `dst_index`; the result has the
    same type as the input tile.
  }];
  let arguments = (ins TTL_TileType:$input, F32Attr:$value, Index:$dst_index);
  let results = (outs TTL_TileType:$result);
  let assemblyFormat =
      "$input `,` $value `into` `dst` `[` $dst_index `]` attr-dict `:` "
      "type($input) `->` type($result)";
}

def TTL_MulUnaryConstOp : TTL_ComputePrimitiveOp<
    "mul_unary_const", "MultiplyByConstant",
    [Pure, TTL_UnaryElementwiseOpTrait,
     DeclareOpInterfaceMethods<TTL_DFBInputOpInterface>,
     AllTypesMatch<["input", "result"]>]> {
  let summary = "Multiply a tensor of tiles by a compile-time scalar constant";
  let description = [{
    Element-wise multiply of an input tensor of tiles by a compile-time
    scalar constant. The result has the same type as the input.
  }];
  let arguments = (ins AnyRankedTensor:$input, F32Attr:$value);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat =
      "$input `,` $value attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Matrix Multiplication
//===----------------------------------------------------------------------===//

def TTL_MatmulOp : TTL_ComputePrimitiveOp<"matmul", "Matmul", [Pure,
    DeclareOpInterfaceMethods<TTL_DFBInputOpInterface>]> {
  let summary = "Matrix multiply two tensors of tiles";
  let description = [{
    Block-level matrix multiply: result[i, j] = sum_k(lhs[i, k] * rhs[k, j]).
    Both inputs must be rank-2 tensors of tiles. The inner (K) dimensions
    must match: lhs.shape[1] == rhs.shape[0]. Result shape is
    [lhs.shape[0], rhs.shape[1]].

    When the optional `transpose_rhs` unit attribute is set, `rhs` is the
    transpose of B and stored as [N, K]: result[i, j] = sum_k(lhs[i, k] *
    rhs[j, k]). The inner (K) dimensions then match as lhs.shape[1] ==
    rhs.shape[1], and the result shape is [lhs.shape[0], rhs.shape[0]].

    Operand tile dimensions follow the same matrix relation. Without
    `transpose_rhs`, `lhs.tile.width == rhs.tile.height` and the result tile is
    `[lhs.tile.height, rhs.tile.width]`. With `transpose_rhs`, the rhs tile
    width is contracted and the result tile width is the rhs tile height.
    The supported element data type combinations are equal lhs, rhs, and result
    types, or BF16 lhs and result types with a BFP4_B or BFP8_B rhs. The
    mixed-format form does not support rhs transposition.
  }];
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs,
                       UnitAttr:$transpose_rhs);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

def TTL_TileMatmulBlockOp : TTL_TileOp<"tile_matmul_block", "Matmul",
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

    When the optional `transpose_rhs` unit attribute is set, the B operand
    is the transpose of the multiplicand and stored as [N, K]; the attribute
    is forwarded to matmul_block's transpose argument and selects the
    transposed per-K addressing during TTKernel lowering.

    Maps to ttkernel.mm_block_init + ttkernel.matmul_block.

    The lhs, rhs, and result tile types may have different dimensions. Their
    physical dimensions follow the same contraction relation as `ttl.matmul`.
    The supported mixed-format form is BF16 lhs by BFP4_B or BFP8_B rhs with
    BF16 output and no rhs transpose. Destination accumulation precision
    remains a compute-kernel configuration property.
  }];
  let arguments = (ins TTL_TileOrTensor:$lhs, TTL_TileOrTensor:$rhs,
                       Optional<TTL_TileOrTensor>:$accumulator,
                       Index:$dst_index, UnitAttr:$transpose_rhs);
  let results = (outs TTL_TileOrTensor:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs (`,` $accumulator^)? `into` `dst` `[` $dst_index `]` attr-dict `:`
    type($lhs) `,` type($rhs) (`,` type($accumulator)^)? `->` type($result)
  }];
  let hasVerifier = 1;
}

// Circular buffer synchronization operations
//===----------------------------------------------------------------------===//

class TTL_DFBLifecycleOp<string mnemonic, I32EnumAttrCase effectKind,
                         list<Trait> traits = []>
    : TTL_Op<mnemonic,
             !listconcat(traits,
                         [DeclareOpInterfaceMethods<TTL_DFBAccessOpInterface>])> {
  let extraClassDefinition = [{
    ::llvm::SmallVector<::mlir::Value>
    $cppClass::getDFBDependencyOperands() {
      return {getCb()};
    }

    ::llvm::SmallVector<::mlir::tt::ttl::DFBProtocolEffect>
    $cppClass::getDFBProtocolEffects() {
      ::mlir::Value dfb = getCb();
      auto explicitTileCount = getNumTiles();
      int64_t tileCount =
          explicitTileCount
              ? static_cast<int64_t>(*explicitTileCount)
              : ::mlir::cast<::mlir::tt::ttl::CircularBufferType>(dfb.getType())
                    .getElementsPerBlock();
      return {{dfb, ::mlir::tt::ttl::DFBProtocolEffectKind::}] #
      effectKind.symbol # [{, tileCount, 0, 0}};
    }
  }];
}

def TTL_CBReserveOp
    : TTL_DFBLifecycleOp<"cb_reserve", TTL_DFBProtocolEffectKind_Reserve,
        [MemoryEffects<[MemWrite]>,
         DeclareOpInterfaceMethods<ViewLikeOpInterface>]> {
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
    ConfinedAttr<OptionalAttr<I64Attr>, [IntPositive]>:$num_tiles
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$cb attr-dict `:` type($cb) `->` type($result)";
  let hasVerifier = 1;
}

def TTL_CBPushOp
    : TTL_DFBLifecycleOp<"cb_push", TTL_DFBProtocolEffectKind_Push,
                        [MemoryEffects<[MemWrite]>]> {
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
    ConfinedAttr<OptionalAttr<I64Attr>, [IntPositive]>:$num_tiles
  );
  let assemblyFormat = "$cb attr-dict `:` type($cb)";
  let hasVerifier = 1;
}

def TTL_CBWaitOp
    : TTL_DFBLifecycleOp<"cb_wait", TTL_DFBProtocolEffectKind_Wait,
                        [DeclareOpInterfaceMethods<ViewLikeOpInterface>]> {
  let summary = "Wait for pages to be available in the circular buffer";
  let description = [{
    `ttl.cb_wait` blocks until pages are available in the circular buffer for
    reading. Returns a tensor view of the available pages.

    This operation is used by consumer threads (typically compute kernels or
    data movement kernels writing to DRAM) to wait for data from producers.
    The synchronization remains observable when the returned view has no SSA
    uses.

    The number of pages defaults to the CB's elements per block, but can be
    overridden via the optional `num_tiles` attribute when the wait spans
    multiple coalesced blocks (see `ttl-coalesce-dfb-acquires`).

    Example:
    ```mlir
    %view = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %coalesced = ttl.cb_wait %cb {num_tiles = 3 : i64} : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x3x!ttcore.tile<32x32, bf16>>
    ```
  }];
  let arguments = (ins
    TTL_CircularBuffer:$cb,
    ConfinedAttr<OptionalAttr<I64Attr>, [IntPositive]>:$num_tiles
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$cb attr-dict `:` type($cb) `->` type($result)";
  let hasVerifier = 1;
}

def TTL_CBPopOp
    : TTL_DFBLifecycleOp<"cb_pop", TTL_DFBProtocolEffectKind_Pop,
                        [MemoryEffects<[MemWrite]>]> {
  let summary = "Release pages from the circular buffer after reading";
  let description = [{
    `ttl.cb_pop` signals that pages have been consumed from the circular buffer
    and the space can be reused by producers.

    This operation must be called after reading data acquired via `ttl.cb_wait`.
    It increments the CB's consumer pointer.

    The number of pages defaults to the CB's elements per block, but can be
    overridden via the optional `num_tiles` attribute when the pop releases
    multiple coalesced blocks (see `ttl-coalesce-dfb-acquires`).

    Example:
    ```mlir
    ttl.cb_pop %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb {num_tiles = 3 : i64} : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    ```
  }];
  let arguments = (ins
    TTL_CircularBuffer:$cb,
    ConfinedAttr<OptionalAttr<I64Attr>, [IntPositive]>:$num_tiles
  );
  let assemblyFormat = "$cb attr-dict `:` type($cb)";
  let hasVerifier = 1;
}

def TTL_StoreOp : TTL_ComputePrimitiveOp<"store", "Passthrough",
    [MemoryEffects<[MemWrite]>]> {
  let summary = "Store a tensor to an acquired dataflow buffer view";
  let description = [{
    `ttl.store` represents the user's intent to store a computed tensor into
    an acquired dataflow buffer view. A reserve-backed destination writes a
    producer-owned slot. A wait-backed destination replaces a consumer-owned
    slot after the original value has been consumed.

    Python's `o.store(result)` emits an overwrite. `o += result` emits packer
    accumulation for a reserve-backed destination or an explicit
    read-add-replacement sequence for a wait-backed destination.

    Wait-backed replacement is valid only when compiler analysis proves that
    the one-block DFB is acquired completely, so the read and write pointers
    address the same pages. It does not advance either pointer and the existing
    `ttl.cb_pop` remains the release operation. The replacement must execute
    after a read of the acquired value, and its wait must be the first access
    to that DFB in the compute kernel. Multi-block, partial-ring, conditional,
    repeated, and otherwise unresolved mutations are rejected.

    When `accumulate` is set, the destination must be reserve-backed. The
    enclosing loop is annotated for L1
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

//===----------------------------------------------------------------------===//
// Raw element access operations (data movement threads only)
//===----------------------------------------------------------------------===//

def TTL_RawElementReadOp : TTL_Op<"raw_element_read",
    [MemoryEffects<[MemRead]>, TTL_DataMovementOpTrait]> {
  let summary = "Read a scalar element from a block at flat coordinates";
  let description = [{
    `ttl.raw_element_read` extracts a single scalar element from a block
    (a tensor view acquired from `ttl.cb_wait` or `ttl.cb_reserve`) at
    the given coordinates. Coordinates are flat scalar-element positions
    regardless of layout: for tiled blocks, lowering decomposes them into
    tile + intra-tile offsets; for row-major blocks, they index directly.

    The result type must match the block's underlying scalar element type
    (f32 for Float32, bf16 for BFloat16). Only f32 and bf16 are currently
    supported.

    This operation is restricted to data movement (noc) threads.

    Example:
    ```mlir
    %val = ttl.raw_element_read %block[%i, %j]
           : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
    ```
  }];
  let arguments = (ins
    AnyRankedTensor:$block,
    Variadic<Index>:$coords
  );
  let results = (outs AnyTypeOf<[F32, BF16]>:$result);
  let assemblyFormat =
      "$block `[` $coords `]` attr-dict `:` type($block) `->` type($result)";
  let hasVerifier = 1;
}

def TTL_ReadIndexOp : TTL_Op<"read_index",
    [MemoryEffects<[MemRead]>, TTL_DataMovementOpTrait]> {
  let summary = "Read a nonnegative scalar element as an index";
  let description = [{
    `ttl.read_index` reads a single f32 or bf16 scalar element from a block
    acquired from a dataflow buffer by `ttl.cb_wait` and converts it to an
    index. Coordinates are flat scalar-element positions and follow the same
    layout-independent rules as `ttl.raw_element_read`.

    Conversion truncates fractional values toward zero. The source value must
    be finite, nonnegative, and no greater than INT32_MAX; behavior is
    undefined otherwise. This operation is intended for tensor-provided
    offsets such as table indices and positions.

    This operation is restricted to data movement (noc) threads.

    Example:
    ```mlir
    %index = ttl.read_index %block[%row, %column]
             : tensor<1x1x!ttcore.tile<32x32, f32>> -> index
    ```
  }];
  let arguments = (ins
    AnyRankedTensor:$block,
    Variadic<Index>:$coords
  );
  let results = (outs Index:$result);
  let assemblyFormat =
      "$block `[` $coords `]` attr-dict `:` type($block) `->` type($result)";
  let hasVerifier = 1;
}

def TTL_RawElementWriteOp : TTL_Op<"raw_element_write",
    [MemoryEffects<[MemRead, MemWrite]>, TTL_DataMovementOpTrait]> {
  let summary = "Write a scalar value to a block at flat coordinates";
  let description = [{
    `ttl.raw_element_write` stores a single scalar value into a block
    (a tensor view acquired from `ttl.cb_wait` or `ttl.cb_reserve`) at
    the given coordinates. Coordinates are flat scalar-element positions
    regardless of layout: for tiled blocks, lowering decomposes them into
    tile + intra-tile offsets; for row-major blocks, they index directly.

    The value type must match the block's underlying scalar element type
    (f32 for Float32, bf16 for BFloat16). Only f32 and bf16 are currently
    supported.

    This operation is restricted to data movement (noc) threads.

    Example:
    ```mlir
    ttl.raw_element_write %block[%i, %j], %val
           : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
    ```
  }];
  let arguments = (ins
    AnyRankedTensor:$block,
    Variadic<Index>:$coords,
    AnyTypeOf<[F32, BF16]>:$value
  );
  let assemblyFormat =
      "$block `[` $coords `]` `,` $value attr-dict `:` type($block) `,` type($value)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Tensor raw address
//===----------------------------------------------------------------------===//

def TTL_RawAddrOp : TTL_Op<"raw_addr", [Pure]> {
  let summary = "Get raw i32 base address for a tensor argument";
  let description = [{
    `ttl.raw_addr` returns the raw 32-bit base address associated with a
    tensor function argument. This bypasses TensorAccessor materialization and
    is intended for external-function interop where a raw pointer/address is
    required.

    Every kernel that references the tensor receives its buffer base as a common
    runtime argument. Unlike a tensor operand to `ttl.opaque_call`, this op does
    not materialize a `TensorAccessor` or require its compile-time arguments.
    The result carries no tensor layout, distribution, page size, alignment,
    extent, or identity metadata. Slice/view operands are unsupported.
  }];
  let arguments = (ins TTLLayoutTensor:$tensor);
  let results = (outs I32:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Synchronized dataflow buffer reset
//===----------------------------------------------------------------------===//

def TTL_ResetDFBsOp : TTL_Op<"reset_dfbs", [MemoryEffects<[MemWrite]>]> {
  let summary = "Synchronize DFB owners and reset selected interfaces";
  let description = [{
    `ttl.reset_dfbs` is one worker-local synchronization boundary shared by
    the three logical kernels named by `reset`. Each participant executes the
    same zero-or-one dynamic instance per dispatch and launch node, or one
    instance per iteration of an equivalent immutable sequential loop nest.
    Before any participant returns, the operation completes prior DFB-interface
    work and restores the listed DFBs to canonical empty protocol state without
    changing descriptor configuration or payload storage.

    The operation resets read and write pointers, the packer write-tile
    pointer, initialization state, and stream occupancy counters. It does not
    clear payload bytes or change the descriptor. Each participating data
    movement RISC drains its own outstanding NoC commands before publishing
    boundary arrival. The operation cannot complete commands issued by another
    core or a non-participating RISC, so every producer must issue its required
    transfers before its local boundary occurrence. Runtime lowering is
    supported only for Blackhole.
  }];
  let arguments = (ins
    TTL_SynchronizedDFBResetAttr:$reset,
    Variadic<TTL_CircularBuffer>:$dfbs
  );
  let assemblyFormat = "$reset `(` $dfbs `:` type($dfbs) `)` attr-dict";
  let hasVerifier = 1;
}

def TTL_ResetAllDFBsOp
    : TTL_Op<"reset_all_dfbs", [MemoryEffects<[MemWrite]>]> {
  let summary = "Synchronize DFB owners and reset every local interface";
  let description = [{
    `ttl.reset_all_dfbs` has the synchronization and state-reset semantics of
    `ttl.reset_dfbs`, applied to every worker-local DFB allocated by the
    program. It does not take DFB operands because the complete allocation set
    is resolved after physical DFB indices are finalized.
  }];
  let arguments = (ins TTL_SynchronizedDFBResetAttr:$reset);
  let assemblyFormat = "$reset attr-dict";
}

//===----------------------------------------------------------------------===//
// Dataflow-buffer reconfiguration boundary
//===----------------------------------------------------------------------===//

def TTL_DFBReconfigurationOp : TTL_Op<"dfb_reconfiguration",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Enter the next worker-local DFB configuration epoch";
  let description = [{
    `ttl.dfb_reconfiguration` synchronizes the logical kernels named by
    `boundary` before installing the next DFB configuration. On each launch
    node, all participants execute the operation under equivalent control flow:
    either the same static count and loop trip-count sequence or the same
    conditional execution. Multiple reconfiguration operations execute in the
    same order and declare the same participant set. DFB-interface work before
    each execution completes before installation, and DFB-interface work after
    it begins after installation. Independent math and SFPU work may overlap
    the operation.

    The source operation does not name physical DFB indices or enumerate the
    interfaces that change. Allocation analysis derives the configuration for
    each epoch from logical DFB lifetimes. Live protocol state retains its
    physical index, storage, and interface configuration across the operation;
    reassignment requires a complete protocol and synchronization proof unless
    the operation declares that unread producer payload and opaque external
    protocol state may be discarded.

    Runtime execution is currently supported only on Blackhole. Compilation
    rejects a resolved non-Blackhole target. Shared-L1 synchronization state
    coordinates interface changes across the participating kernels.
  }];
  let arguments = (ins TTL_DFBReconfigurationAttr:$boundary);
  let assemblyFormat = "$boundary attr-dict";
}

//===----------------------------------------------------------------------===//
// External function call (opaque)
//===----------------------------------------------------------------------===//

def TTL_OpaqueCallOp : TTL_Op<"opaque_call",
    [MemoryEffects<[MemRead, MemWrite]>, AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<TTL_DFBAccessOpInterface,
                               ["getDFBNonTransactionalAccesses",
                                "getDFBIndexOperands",
                                "hasUnknownDFBAccess"]>]> {
  let summary = "Call an external C/C++ function by name";
  let description = [{
    `ttl.opaque_call` emits a call to an external C/C++ function whose
    declaration is provided by a user-specified header file. The callee
    name and header path are string attributes; the call is opaque to the
    compiler (no inlining, no type checking of the foreign signature).

    The `header` attribute specifies the path to the C/C++ header that
    declares the callee. The lowering pass emits a `#include` directive
    for this header in the generated kernel source.

    The ordered `template_args` attributes become C++ template arguments.
    Scalar entries contain their complete compile-time payload. DFB index and
    descriptor entries reference `template_dfb_operands` by position, which
    preserves DFB identity until allocation is finalized without representing
    scalar metadata as SSA values.

    The `arg_operands` are forwarded as function arguments to the callee. DFB
    arguments and descriptor template operands are dependencies automatically.
    `dependency_dfb_operands` adds dependencies that are not emitted as C++
    arguments. `getDFBDependencyOperands()` returns each DFB function-argument
    occurrence, each descriptor template-argument occurrence, then each
    dependency-only operand. Operand adaptation may map distinct occurrences to
    the same DFB without merging them.

    A base tensor argument is converted according to the enclosing logical
    kernel and its `ttl.layout`. Data-movement kernels receive a
    `TensorAccessor` for device DRAM or SRAM, including ND-sharded storage.
    `L1` and `L1Small` identify the corresponding TTNN SRAM buffer types;
    `L1Small` requires sharded storage.
    Compute kernels receive a byte-addressable `LocalTensorAccessor<uint8_t>`
    only for height-, width-, or block-sharded SRAM. Tensor slices and views are
    invalid. Repeated occurrences of one tensor share one accessor object while
    remaining distinct positional C++ arguments. The callee
    accepts the accessor by `const&` and must not retain its address or
    reference after the enclosing kernel function returns.

    The ordered `dfb_effects` list describes reserve, push, wait, and pop
    actions completed synchronously before the external call returns. Each
    effect's dependency index identifies an element returned by
    `getDFBDependencyOperands()` and carries a positive static tile count no
    greater than the selected DFB's capacity. Every listed effect occurs on
    every execution of the call. Repeated effects retain list order. The summary
    does not emit protocol calls.

    The ordered `dfb_accesses` list describes synchronous DFB accesses that do
    not perform queue transactions. `inspect` states that a callee may read the
    dependency's descriptor or contents but does not publish, consume, or leave
    the DFB changed when it returns. One dependency occurrence cannot declare
    both a protocol effect and an inspection because their relative execution
    order would be unspecified. Effects and accesses omitted for a dependency
    occurrence leave its access opaque and incomplete for reuse with another
    logical DFB on a shared launch node unless a matching synchronized reset
    proves lifecycle completion. If one DFB occupies multiple dependency
    occurrences, every occurrence requires its own summary or the same reset-
    completion proof. The `unknown_dfb_access` unit attribute declares that the
    callee may also access unlisted user-managed DFBs and adds an opaque
    occurrence to each affected DFB. Exact disjoint launch-node domains remain
    eligible for physical-index sharing.

    A named DFB dependency occurrence without `dfb_effects` or `dfb_accesses`
    contributes an opaque storage access that begins at call entry. Its protocol
    and asynchronous interface work may remain live after the call. When the
    call is ordered before a synchronized DFB reset through the same
    participating logical kernel, reset completion terminates the access and
    canonicalizes its unknown protocol state. The reset implementation must
    complete each participant's earlier interface work before publishing
    arrival. Storage reuse is permitted only after the reset. This does not
    validate the callee's internal queue protocol.

    Every action declared by `dfb_effects` must occur before the call returns.
    Interface work still active after return must remain ordered by the
    declared protocol and complete before its terminal consumer release or a
    synchronized reset. Work that can outlive that terminal must leave the
    affected DFB dependency opaque.

    `unsigned_arg_indices` identifies signless i32 operands that represent
    unsigned addresses at the C++ boundary. The lowering inserts explicit
    unsigned casts so overload resolution and parameter conversion do not
    depend on an incidental signed intermediate type.

    A DFB descriptor entry is itself a direct DFB dependency. A DFB index entry
    does not declare a call-level DFB access. If the callee consumes that index,
    the same DFB must be returned by `getDFBDependencyOperands()`.

    The optional result is a signless i32 or i64 scalar. When absent the call
    is treated as a void function invocation.

    `condition_result` states that the scalar result evaluates one immutable
    dispatch-stable condition. Calls using the same typed identity may be
    evaluated independently in different logical kernels. Such a call cannot
    access a DFB, declare DFB effects or non-transactional accesses, or declare
    unknown DFB access because condition evaluation is stable and repeat-safe
    rather than stateful.
    `fabric_manager_effects` declares ownership effects for external fabric
    connection managers. The enclosing operation binds each claim to one
    logical kernel. Nonlocal verification requires balanced ownership and
    rejects uses outside an acquired interval.

    Example:
    ```mlir
    ttl.opaque_call "my_func" () {header = "my_header.hpp"} : () -> ()
    ttl.opaque_call "add_vals" template_args [
        #ttl.external_template_arg<signed_integer, 1>,
        #ttl.external_template_arg<signed_integer, 2>]
        (%a, %b) {header = "math_utils.hpp"} : (i64, i64) -> ()
    ```
  }];
  let arguments = (ins
    StrAttr:$callee,
    StrAttr:$header,
    Variadic<AnyType>:$arg_operands,
    Variadic<TTL_CircularBuffer>:$template_dfb_operands,
    Variadic<TTL_CircularBuffer>:$dependency_dfb_operands,
    OptionalAttr<ArrayAttr>:$template_args,
    OptionalAttr<DenseI32ArrayAttr>:$unsigned_arg_indices,
    OptionalAttr<TypedArrayAttrBase<TTL_FabricManagerEffectAttr,
                                    "fabric manager effects">>:$fabric_manager_effects,
    OptionalAttr<TTL_DFBProtocolEffectArrayAttr>:$dfb_effects,
    OptionalAttr<TTL_DFBNonTransactionalAccessArrayAttr>:$dfb_accesses,
    UnitAttr:$unknown_dfb_access,
    OptionalAttr<TTL_DispatchConditionAttr>:$condition_result
  );
  let results = (outs Optional<AnyTypeOf<[I32, I64]>>:$result);
  let assemblyFormat = [{
    $callee (`template_args` $template_args^)?
    (`template_dfbs` `(` $template_dfb_operands^ `:`
     type($template_dfb_operands) `)`)?
    (`dfb_dependencies` `(` $dependency_dfb_operands^ `:`
     type($dependency_dfb_operands) `)`)?
    (`dfb_effects` $dfb_effects^)?
    (`dfb_accesses` $dfb_accesses^)?
    ` ` `(` $arg_operands `)` attr-dict `:`
    functional-type($arg_operands, results)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DFB index query
//===----------------------------------------------------------------------===//

def TTL_GetDfbIdOp : TTL_Op<"get_dfb_id", [Pure]> {
  let summary = "Get the integer index for a dataflow buffer";
  let description = [{
    Returns the integer DFB index for the given dataflow buffer as an
    i32 SSA value. At EmitC time this resolves to a raw integer literal.
    Integer SSA dataflow does not preserve a call-level DFB access; an
    external callee that accesses the DFB must also receive it as a DFB
    operand.
  }];
  let arguments = (ins TTL_CircularBuffer:$dfb);
  let results = (outs I32:$result);
  let assemblyFormat = "$dfb attr-dict `:` type($dfb)";
}

def TTL_TileStoreOp : TTL_ComputePrimitiveOp<"tile_store", "Passthrough",
    [MemoryEffects<[MemWrite]>, TTL_DstResultOpTrait,
     TTL_DstAccessOpInterface, TTL_TileExecutionOpInterface]> {
  let summary = "Store a tile into an acquired dataflow buffer view";
  let description = [{
    `ttl.tile_store` packs a single tile from the DST register into a dataflow
    buffer view obtained from `ttl.cb_reserve` or `ttl.cb_wait`. This is the
    tile-level form of `ttl.store`.

    A `producer` store packs into producer-reserved pages. A
    `consumer_replacement` store packs into waited pages without changing
    occupancy or either dataflow buffer pointer. The latter is valid only when
    the original generation has been read, the complete one-block DFB is
    acquired, and all replacement reads precede the matching pop. `store_kind`
    preserves this typed ownership decision when the view is replaced or
    sliced.

    The `indices` operands are multi-dimensional DFB tile coordinates. The
    `dst_index` operand identifies the source tile in the DST register.

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
    Index:$dst_index,
    DefaultValuedAttr<TTL_DFBTileStoreKindAttr,
      "::mlir::tt::ttl::DFBTileStoreKind::Producer">:$store_kind
  );
  let assemblyFormat = "$tile `,` $view `[` $indices `]` `from` `dst` `[` $dst_index `]` attr-dict `:` type($tile) `,` type($view)";
  let hasVerifier = 1;
}

#endif // TTLANG_DIALECT_TTL_IR_TTLOPS_TD
