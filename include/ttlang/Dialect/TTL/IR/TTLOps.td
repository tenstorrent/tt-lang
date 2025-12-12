// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPS_TD
#define TTLANG_DIALECT_TTL_IR_TTLOPS_TD

include "ttlang/Dialect/TTL/IR/TTLBase.td"
include "ttlang/Dialect/TTL/IR/TTLOpsTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TTL operation definitions
//===----------------------------------------------------------------------===//

def TTL_CreateCBOp : TTL_Op<"create_cb", [Pure]> {
  let summary = "Create a circular buffer value";
  let arguments = (ins
    I64ArrayAttr:$shape,
    TypeAttr:$element_type,
    I64Attr:$buffer_factor,
    OptionalAttr<I32Attr>:$buffer_index,
    OptionalAttr<I64Attr>:$page_size
  );
  let results = (outs TTL_CircularBuffer:$result);
  let assemblyFormat = "`(` `)` attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def TTL_CopyOp : TTL_Op<"copy", [Pure]> {
  let summary = "Asynchronous copy between tensor and circular buffer";
  let arguments = (ins
    AnyType:$src,
    AnyType:$dst
  );
  let results = (outs TTL_TransferHandle:$xf);
  let assemblyFormat = "$src `,` $dst attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TTL_WaitOp : TTL_Op<"wait", [Pure]> {
  let summary = "Wait on a transfer handle";
  let arguments = (ins TTL_TransferHandle:$xf);
  let assemblyFormat = "$xf attr-dict";
}

#endif // TTLANG_DIALECT_TTL_IR_TTLOPS_TD
