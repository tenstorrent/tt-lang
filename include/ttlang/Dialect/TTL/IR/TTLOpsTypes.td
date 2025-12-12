// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPSTYPES_TD
#define TTLANG_DIALECT_TTL_IR_TTLOPSTYPES_TD

include "ttlang/Dialect/TTL/IR/TTLBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

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

def TTL_TransferHandle : TTL_Type<"TransferHandle", "xf"> {
  let summary = "Handle for asynchronous TTL memory transfers";
}

#endif // TTLANG_DIALECT_TTL_IR_TTLOPSTYPES_TD
