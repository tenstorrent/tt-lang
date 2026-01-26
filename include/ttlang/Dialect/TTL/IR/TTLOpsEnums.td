// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPSENUMS_TD
#define TTLANG_DIALECT_TTL_IR_TTLOPSENUMS_TD

include "ttlang/Dialect/TTL/IR/TTLBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// TTL enum definitions
//===----------------------------------------------------------------------===//

// Note: TTL uses TTKernel's ThreadType enum (#ttkernel.thread<noc>) for
// kernel thread annotations on functions.

def TTL_TransferKind : I32EnumAttr<
    "TransferKind", "TTL transfer kind",
    [
      I32EnumAttrCase<"read", 0>,
      I32EnumAttrCase<"write", 1>,
    ]> {
  let cppNamespace = "::mlir::tt::ttl";
  let genSpecializedAttr = 0;
}

def TTL_BcastDim : I32EnumAttr<
    "BcastDim", "TTL broadcast dimension",
    [
      I32EnumAttrCase<"row", 0>,    // Broadcast along rows (input is Nx1)
      I32EnumAttrCase<"col", 1>,    // Broadcast along columns (input is 1xN)
      I32EnumAttrCase<"scalar", 2>, // Scalar broadcast (input is 1x1)
    ]> {
  let cppNamespace = "::mlir::tt::ttl";
}

def TTL_BcastDimAttr : EnumAttr<TTL_Dialect, TTL_BcastDim, "bcast_dim">;

#endif // TTLANG_DIALECT_TTL_IR_TTLOPSENUMS_TD
