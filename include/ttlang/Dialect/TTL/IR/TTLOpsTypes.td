// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPSTYPES_TD
#define TTLANG_DIALECT_TTL_IR_TTLOPSTYPES_TD

include "ttlang/Dialect/TTL/IR/TTLBase.td"
include "ttlang/Dialect/TTL/IR/TTLOpsEnums.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

// A ranked tensor with TTL layout encoding. This constraint is useful for
// operations that require tensor tile/addressing info derived from LayoutAttr.
def TTLLayoutTensor : TypeConstraint<
    And<[
      CPred<"llvm::isa<mlir::RankedTensorType>($_self)">,
      CPred<"llvm::cast<mlir::RankedTensorType>($_self).getEncoding() && "
            "llvm::isa<mlir::tt::ttl::LayoutAttr>("
            "llvm::cast<mlir::RankedTensorType>($_self).getEncoding())">
    ]>,
    "ranked tensor with TTL layout encoding">;

//===----------------------------------------------------------------------===//
// TTL type definitions
//===----------------------------------------------------------------------===//

def TTL_CircularBuffer : TTL_Type<"CircularBuffer", "cb"> {
  let summary = "Circular buffer in L1 for TT-Lang kernels";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType,
    "int64_t":$blockCount
  );
  let assemblyFormat = "`<` `[` $shape `]` `,` $elementType `,` $blockCount `>`";

  let extraClassDeclaration = [{
    int64_t getElementsPerBlock() const {
      return std::accumulate(getShape().begin(), getShape().end(), int64_t{1},
                             std::multiplies<int64_t>());
    }

    int64_t getTotalElements() const {
      return getElementsPerBlock() * getBlockCount();
    }
  }];
}

def TTL_TransferHandle : TTL_Type<"TransferHandle", "transfer_handle"> {
  let summary = "Handle for asynchronous TTL memory transfers";
  let parameters = (ins
    OptionalParameter<"std::optional<mlir::tt::ttl::TransferKind>">:$kind
  );
  let assemblyFormat = "(`<` $kind^ `>`)?";
}

def TTL_PipeTransfer : TTL_Type<"PipeTransfer", "pipe_transfer"> {
  let summary = "Pipe transfer contract";
  let description = [{
    Represents one logical receiver-posted pipe transfer before physical
    synchronization resources are assigned. The transfer value identifies the
    source/receiver relation and the point-to-point or collective contract; it
    is not a TTKernel semaphore, SRAM table, GlobalSemaphore, or DFB slot.
  }];
}

def TTL_PipeToken : TTL_Type<"PipeToken", "pipe_token"> {
  let summary = "Pipe receive completion token";
  let description = [{
    Names the receiver-side completion dependency produced by a pipe receive
    post. Waiting on the token makes the receiver-owned destination DFB slot
    safe to consume.
  }];
  let parameters = (ins
    "int64_t":$pipeNetId
  );
  let assemblyFormat = "`<` `net` $pipeNetId `>`";
}

def TTL_DSTRegister : TTL_Type<"DSTRegister", "dst"> {
  let summary = "Handle representing a tile resident in a DST register";
  let description = [{
    A lightweight handle (token) indicating that a tile now lives in a DST register
    slot. Carries no payload; downstream consumers use it to ensure correct
    ordering of DST-producing/consuming operations.
  }];
}

//===----------------------------------------------------------------------===//
// Enum attribute wrappers
//===----------------------------------------------------------------------===//

def TTL_BcastTypeAttr : EnumAttr<TTL_Dialect, TTL_BcastType, "bcast_type">;
def TTL_ReduceTypeAttr : EnumAttr<TTL_Dialect, TTL_ReduceType, "reduce_type">;

def TTL_Pipe : TTL_Type<"Pipe", "pipe"> {
  let summary = "Pipe for core-to-core data transfer";
  let description = [{
    A pipe represents a transfer relation from a source core to one or more
    receiver cores. Used with `ttl.copy` for inter-core data movement.

    For point-to-point transfers, dstStart == dstEnd. For collective
    transfers, dstStart and dstEnd define the receiver-set corners.

    Example:
    ```mlir
    // Point-to-point pipe from (0,0) to (1,0)
    %p = ttl.create_pipe src(0, 0) dst(1, 0) : !ttl.pipe

    // Collective pipe from (0,0) to receiver set (1,0)-(1,3)
    %p = ttl.create_pipe src(0, 0) dst(1, 0) to (1, 3) : !ttl.pipe
    ```
  }];
  let parameters = (ins
    "int64_t":$srcX,
    "int64_t":$srcY,
    "int64_t":$dstStartX,
    "int64_t":$dstStartY,
    "int64_t":$dstEndX,
    "int64_t":$dstEndY,
    "int64_t":$pipeNetId
  );
  let assemblyFormat = "`<` `src` `(` $srcX `,` $srcY `)` `dst` `(` $dstStartX `,` $dstStartY `)` `to` `(` $dstEndX `,` $dstEndY `)` `net` $pipeNetId `>`";

  let extraClassDeclaration = [{
    /// Returns true if this pipe has one receiver in its destination extent.
    bool hasSingleReceiver() const {
      return getDstStartX() == getDstEndX() && getDstStartY() == getDstEndY();
    }

    /// Returns true if this pipe has more than one receiver.
    bool hasMultipleReceivers() const { return !hasSingleReceiver(); }

    LLVM_DEPRECATED("Use hasSingleReceiver instead", "hasSingleReceiver")
    bool isUnicast() const { return hasSingleReceiver(); }

    LLVM_DEPRECATED("Use hasMultipleReceivers instead", "hasMultipleReceivers")
    bool isMulticast() const { return hasMultipleReceivers(); }

    /// Returns the number of destination cores.
    int64_t getNumDests() const {
      int64_t width = std::abs(getDstEndX() - getDstStartX()) + 1;
      int64_t height = std::abs(getDstEndY() - getDstStartY()) + 1;
      return width * height;
    }

    /// Returns true if the source core is within the destination range.
    bool srcInDstRange() const {
      int64_t minX = std::min(getDstStartX(), getDstEndX());
      int64_t maxX = std::max(getDstStartX(), getDstEndX());
      int64_t minY = std::min(getDstStartY(), getDstEndY());
      int64_t maxY = std::max(getDstStartY(), getDstEndY());
      return getSrcX() >= minX && getSrcX() <= maxX &&
             getSrcY() >= minY && getSrcY() <= maxY;
    }
  }];
}

#endif // TTLANG_DIALECT_TTL_IR_TTLOPSTYPES_TD
