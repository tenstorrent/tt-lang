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

// Thread type for ttl.kernel regions. Maps to TTKernel's ThreadType during
// lowering (compute -> Compute, datamovement -> Noc).
def TTL_ThreadType : I32EnumAttr<
    "ThreadType", "TTL kernel thread type",
    [
      I32EnumAttrCase<"compute", 0>,
      I32EnumAttrCase<"datamovement", 1>,
    ]> {
  let cppNamespace = "::mlir::tt::ttl";
  let genSpecializedAttr = 0;
}

def TTL_ThreadTypeAttr : EnumAttr<TTL_Dialect, TTL_ThreadType, "thread_type">;

def TTL_TransferKind : I32EnumAttr<
    "TransferKind", "TTL transfer kind",
    [
      I32EnumAttrCase<"read", 0>,
      I32EnumAttrCase<"write", 1>,
    ]> {
  let cppNamespace = "::mlir::tt::ttl";
  let genSpecializedAttr = 0;
}

#endif // TTLANG_DIALECT_TTL_IR_TTLOPSENUMS_TD
