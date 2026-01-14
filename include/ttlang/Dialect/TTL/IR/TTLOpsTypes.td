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
// Enum Attributes
//===----------------------------------------------------------------------===//

def TTL_ReduceDimAttr : EnumAttr<TTL_Dialect, TTL_ReduceDim, "reduce_dim"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // TTLANG_DIALECT_TTL_IR_TTLOPSTYPES_TD
