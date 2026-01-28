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

// A ranked tensor with TTNN layout encoding. This constraint is useful for
// operations that require tensor tile/addressing info derived from TTNNLayoutAttr.
def TTNNTensor : TypeConstraint<
    And<[
      CPred<"llvm::isa<mlir::RankedTensorType>($_self)">,
      CPred<"llvm::cast<mlir::RankedTensorType>($_self).getEncoding() && "
            "llvm::isa<mlir::tt::ttnn::TTNNLayoutAttr>("
            "llvm::cast<mlir::RankedTensorType>($_self).getEncoding())">
    ]>,
    "ranked tensor with TTNN layout encoding">;

//===----------------------------------------------------------------------===//
// TTL type definitions
//===----------------------------------------------------------------------===//

def TTL_CircularBuffer : TTL_Type<"CircularBuffer", "cb"> {
  let summary = "Circular buffer in L1 for TT-Lang kernels";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType,
    "int64_t":$bufferFactor
  );
  let assemblyFormat = "`<` `[` $shape `]` `,` $elementType `,` $bufferFactor `>`";

  let extraClassDeclaration = [{
    int64_t getElementsPerBlock() const {
      return std::accumulate(getShape().begin(), getShape().end(), int64_t{1},
                             std::multiplies<int64_t>());
    }

    int64_t getTotalElements() const {
      return getElementsPerBlock() * getBufferFactor();
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
def TTL_ReduceDimAttr : EnumAttr<TTL_Dialect, TTL_ReduceDim, "reduce_dim">;

def TTL_Pipe : TTL_Type<"Pipe", "pipe"> {
  let summary = "Pipe for core-to-core data transfer";
  let description = [{
    A pipe represents a communication channel from a source core to one or more
    destination cores. Used with `ttl.copy` for inter-core data movement.

    For unicast (single destination), dstStart == dstEnd.
    For multicast (rectangular grid), dstStart and dstEnd define the corners.

    Example:
    ```mlir
    // Unicast pipe from (0,0) to (1,0)
    %p = ttl.create_pipe src(0, 0) dst(1, 0) : !ttl.pipe

    // Multicast pipe from (0,0) to rectangular region (1,0)-(1,3)
    %p = ttl.create_pipe src(0, 0) dst(1, 0) to (1, 3) : !ttl.pipe
    ```
  }];
  let parameters = (ins
    "int64_t":$srcX,
    "int64_t":$srcY,
    "int64_t":$dstStartX,
    "int64_t":$dstStartY,
    "int64_t":$dstEndX,
    "int64_t":$dstEndY
  );
  let assemblyFormat = "`<` `src` `(` $srcX `,` $srcY `)` `dst` `(` $dstStartX `,` $dstStartY `)` `to` `(` $dstEndX `,` $dstEndY `)` `>`";

  let extraClassDeclaration = [{
    /// Returns true if this is a unicast pipe (single destination).
    bool isUnicast() const {
      return getDstStartX() == getDstEndX() && getDstStartY() == getDstEndY();
    }

    /// Returns true if this is a multicast pipe (multiple destinations).
    bool isMulticast() const { return !isUnicast(); }

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
