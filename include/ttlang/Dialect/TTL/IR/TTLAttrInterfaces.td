// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLATTRINTERFACES_TD
#define TTLANG_DIALECT_TTL_IR_TTLATTRINTERFACES_TD

include "mlir/IR/Interfaces.td"

def TTL_StructuredTransferAttrInterface :
    AttrInterface<"StructuredTransferAttrInterface"> {
  let description = [{
    Interface for compact transfer relations over one component of a logical
    device domain.
  }];
  let cppNamespace = "::mlir::tt::ttl";
  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the referenced device-domain component.",
      /*retTy=*/"::mlir::StringAttr",
      /*methodName=*/"getComponent",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return $_attr.getComponent(); }]>
  ];
}

#endif // TTLANG_DIALECT_TTL_IR_TTLATTRINTERFACES_TD
