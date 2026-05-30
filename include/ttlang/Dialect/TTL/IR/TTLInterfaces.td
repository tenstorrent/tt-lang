// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLINTERFACES_TD
#define TTLANG_DIALECT_TTL_IR_TTLINTERFACES_TD

include "mlir/IR/OpBase.td"

def TTL_DFBInputOpInterface : OpInterface<"DFBInputOpInterface"> {
  let description = [{
    Tensor-level ops whose tile-level lowerings may read one or more
    operands from dataflow buffers via the unpacker. The set of operands
    that require DFB attachment is computed per instance because it can
    depend on the producer of each operand (e.g., an elementwise consumer
    needs DFB materialization only when its operand comes from a producer
    that cannot fuse with it). Returning an empty list is valid and means
    no operand of this instance needs to be DFB-attached.
  }];

  let cppNamespace = "::mlir::tt::ttl";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the operand indices of this instance that require "
               "DFB-attached values. May be empty.",
      /*retTy=*/"::llvm::SmallVector<unsigned>",
      /*methodName=*/"getDFBInputOperandIndices",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/""
    >
  ];
}

def TTL_PipeNetPredicateOpInterface
    : OpInterface<"PipeNetPredicateOpInterface"> {
  let description = [{
    Predicate ops (`ttl.is_src`, `ttl.is_dst`, `ttl.is_active`) whose `i1`
    result evaluates to `true` on a per-PipeNet role-defined node set.
  }];

  let cppNamespace = "::mlir::tt::ttl";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the PipeNet id this predicate references.",
      /*retTy=*/"int64_t",
      /*methodName=*/"getReferencedPipeNetId",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/"Return the PipeNet role this predicate selects.",
      /*retTy=*/"::mlir::tt::ttl::PipeRole",
      /*methodName=*/"getReferencedRole",
      /*args=*/(ins)>,
  ];
}

#endif // TTLANG_DIALECT_TTL_IR_TTLINTERFACES_TD
