// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPS_TD
#define TTLANG_DIALECT_TTL_IR_TTLOPS_TD

include "ttlang/Dialect/TTL/IR/TTLBase.td"
include "ttlang/Dialect/TTL/IR/TTLOpsTypes.td"
include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.td"
include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
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
// Circular buffer synchronization operations
//===----------------------------------------------------------------------===//

def TTL_CBReserveOp : TTL_Op<"cb_reserve", [MemoryEffects<[MemWrite]>]> {
  let summary = "Reserve pages in a circular buffer for writing";
  let description = [{
    `ttl.cb_reserve` blocks until `num_pages` pages are available in the
    circular buffer for writing. Returns a tensor view of the reserved pages.

    This operation is used by producer threads (typically data movement kernels
    reading from DRAM) to acquire space in the CB before writing data.

    The result tensor has shape matching the CB's block shape and element type.
    The num_pages operand must be a positive i32 value.

    Example:
    ```mlir
    %view = ttl.cb_reserve %cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ```
  }];
  let arguments = (ins
    TTL_CircularBuffer:$cb,
    I32:$num_pages
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$cb `,` $num_pages attr-dict `:` type($cb) `->` type($result)";
  let hasVerifier = 1;
}

def TTL_CBPushOp : TTL_Op<"cb_push", [MemoryEffects<[MemWrite]>]> {
  let summary = "Signal that pages have been written to the circular buffer";
  let description = [{
    `ttl.cb_push` signals that `num_pages` pages have been written to the
    circular buffer and are ready for consumer threads to read.

    This operation must be called after writing data to pages acquired via
    `ttl.cb_reserve`. It increments the CB's producer pointer.

    Example:
    ```mlir
    ttl.cb_push %cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ```
  }];
  let arguments = (ins
    TTL_CircularBuffer:$cb,
    I32:$num_pages
  );
  let assemblyFormat = "$cb `,` $num_pages attr-dict `:` type($cb)";
  let hasVerifier = 1;
}

def TTL_CBWaitOp : TTL_Op<"cb_wait", [MemoryEffects<[MemRead]>]> {
  let summary = "Wait for pages to be available in the circular buffer";
  let description = [{
    `ttl.cb_wait` blocks until `num_pages` pages are available in the circular
    buffer for reading. Returns a tensor view of the available pages.

    This operation is used by consumer threads (typically compute kernels or
    data movement kernels writing to DRAM) to wait for data from producers.

    Example:
    ```mlir
    %view = ttl.cb_wait %cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ```
  }];
  let arguments = (ins
    TTL_CircularBuffer:$cb,
    I32:$num_pages
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$cb `,` $num_pages attr-dict `:` type($cb) `->` type($result)";
  let hasVerifier = 1;
}

def TTL_CBPopOp : TTL_Op<"cb_pop", [MemoryEffects<[MemWrite]>]> {
  let summary = "Release pages from the circular buffer after reading";
  let description = [{
    `ttl.cb_pop` signals that `num_pages` pages have been consumed from the
    circular buffer and the space can be reused by producers.

    This operation must be called after reading data acquired via `ttl.cb_wait`.
    It increments the CB's consumer pointer.

    Example:
    ```mlir
    ttl.cb_pop %cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ```
  }];
  let arguments = (ins
    TTL_CircularBuffer:$cb,
    I32:$num_pages
  );
  let assemblyFormat = "$cb `,` $num_pages attr-dict `:` type($cb)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Kernel container operation
//===----------------------------------------------------------------------===//

def TTL_KernelOp : TTL_Op<"kernel", [
    AttrSizedOperandSegments,
    NoTerminator,
    IsolatedFromAbove
  ]> {
  let summary = "Container for TTL kernel threads operating on a device grid";
  let description = [{
    `ttl.kernel` is a container operation that holds regions for each kernel
    thread (compute and data movement). Each region operates on the same set
    of circular buffers derived from the kernel's inputs and outputs.

    The `threads` attribute specifies the thread type for each region. After
    the `ttl-kernel-regions-to-funcs` pass, each region is extracted to a
    separate `func.func` with appropriate `#ttkernel.thread` attributes.

    Each region receives the same CB arguments corresponding to the kernel's
    inputs and outputs. The CB arguments provide the interface for inter-thread
    communication.

    Example:
    ```mlir
    %result = ttl.kernel {grid = #ttcore.grid<1x1>, threads = [#ttl.thread<datamovement>, #ttl.thread<compute>]}
        ins(%lhs, %rhs : tensor<...>, tensor<...>)
        outs(%out : tensor<...>) {
    ^bb0(%lhs_cb: !ttl.cb<...>, %rhs_cb: !ttl.cb<...>, %out_cb: !ttl.cb<...>):
      // Data movement thread: read inputs
    }, {
    ^bb0(%lhs_cb: !ttl.cb<...>, %rhs_cb: !ttl.cb<...>, %out_cb: !ttl.cb<...>):
      // Compute thread: process data
    } : tensor<...>
    ```
  }];
  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    Variadic<AnyRankedTensor>:$outputs,
    TTL_ThreadArrayAttr:$threads,
    TTCore_GridAttr:$grid
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region VariadicRegion<AnyRegion>:$thread_regions);

  let assemblyFormat = [{
    attr-dict
    `ins` `(` $inputs (`:` type($inputs)^)? `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    $thread_regions (`:` type($results)^)?
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Get the thread type for a region by index.
    ThreadType getRegionThreadType(unsigned regionIndex) {
      auto arr = getThreadsAttr().getValue();
      return mlir::cast<ThreadAttr>(arr[regionIndex]).getThreadType();
    }

    /// Get the number of thread regions.
    unsigned getNumThreadRegions() { return getThreadRegions().size(); }
  }];
}

#endif // TTLANG_DIALECT_TTL_IR_TTLOPS_TD
