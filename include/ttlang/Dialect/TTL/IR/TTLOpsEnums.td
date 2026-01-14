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

// Reduce dimension for reduce operations
def TTL_ReduceDim_Row : I32EnumAttrCase<"Row", 0, "row">;
def TTL_ReduceDim_Col : I32EnumAttrCase<"Col", 1, "col">;
def TTL_ReduceDim_Scalar : I32EnumAttrCase<"Scalar", 2, "scalar">;

def TTL_ReduceDim : I32EnumAttr<
    "ReduceDim", "TTL reduce dimension",
    [TTL_ReduceDim_Row, TTL_ReduceDim_Col, TTL_ReduceDim_Scalar]> {
  let cppNamespace = "::mlir::tt::ttl";
}

#endif // TTLANG_DIALECT_TTL_IR_TTLOPSENUMS_TD
