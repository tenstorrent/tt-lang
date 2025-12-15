// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPSATTRS_TD
#define TTLANG_DIALECT_TTL_IR_TTLOPSATTRS_TD

include "ttlang/Dialect/TTL/IR/TTLBase.td"
include "ttlang/Dialect/TTL/IR/TTLOpsEnums.td"

//===----------------------------------------------------------------------===//
// TTL attribute definitions
//===----------------------------------------------------------------------===//

def TTL_ThreadAttr : TTL_Attr<"Thread", "thread"> {
  let summary = "Thread type attribute for ttl.kernel regions";
  let description = [{
    Annotates a thread region with its type (compute or datamovement).
    After region extraction, includes a symbol reference to the extracted function.
  }];
  let parameters = (ins
    "ThreadType":$threadType,
    OptionalParameter<"mlir::FlatSymbolRefAttr">:$symbolRef
  );
  let assemblyFormat = "`<` $threadType (`,` $symbolRef^)? `>`";
}

def TTL_ThreadArrayAttr : TypedArrayAttrBase<TTL_ThreadAttr, "array of thread attributes">;

def TTL_SliceAttr : TTL_Attr<"Slice", "slice"> {
  let summary = "Slice specification for core range";
  let parameters = (ins
    "int64_t":$start,
    "int64_t":$stop,
    "int64_t":$step
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
  let description = [{
    Encodes a Python slice(start, stop, step) for pipe multicast ranges.
    Half-open interval: [start, stop).
  }];
}

#endif // TTLANG_DIALECT_TTL_IR_TTLOPSATTRS_TD
