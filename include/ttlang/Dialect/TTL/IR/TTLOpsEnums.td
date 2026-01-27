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

def TTL_BcastType_Col    : I32EnumAttrCase<"Col",    1, "col">;
def TTL_BcastType_Row    : I32EnumAttrCase<"Row",    2, "row">;
def TTL_BcastType_Scalar : I32EnumAttrCase<"Scalar", 3, "scalar">;

def TTL_BcastType : I32EnumAttr<"BcastType", "TTL broadcast type",
    [TTL_BcastType_Col, TTL_BcastType_Row, TTL_BcastType_Scalar]> {
  let cppNamespace = "::mlir::tt::ttl";
}

#endif // TTLANG_DIALECT_TTL_IR_TTLOPSENUMS_TD
