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

def TTL_AccumulationScopeOpInterface
    : OpInterface<"AccumulationScopeOpInterface"> {
  let description = [{
    Region-like operations that explicitly describe accumulation semantics.
    The interface separates the accumulation policy from the storage mechanism,
    so consumers can reason about the declared policy directly.
  }];

  let cppNamespace = "::mlir::tt::ttl";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return true when this operation instance carries active "
               "accumulation semantics.",
      /*retTy=*/"bool",
      /*methodName=*/"isAccumulation",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/"Return the destination tensor operands governed by the "
               "accumulation policy.",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getAccumulationOutputs",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/"Return explicit initial values for outputs whose initial mode "
               "requires one.",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getExplicitInitialValues",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/"Return one combiner per accumulation output.",
      /*retTy=*/"::llvm::SmallVector<::mlir::tt::ttl::AccumulationCombiner>",
      /*methodName=*/"getAccumulationCombiners",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/"Return one initial-value mode per accumulation output.",
      /*retTy=*/"::llvm::SmallVector<::mlir::tt::ttl::AccumulationInitialMode>",
      /*methodName=*/"getAccumulationInitialModes",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/"Return the region containing the accumulation body.",
      /*retTy=*/"::mlir::Region &",
      /*methodName=*/"getAccumulationBody",
      /*args=*/(ins)>,
  ];
}

#endif // TTLANG_DIALECT_TTL_IR_TTLINTERFACES_TD
