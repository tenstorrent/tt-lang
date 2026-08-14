// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLINTERFACES_TD
#define TTLANG_DIALECT_TTL_IR_TTLINTERFACES_TD

include "mlir/IR/OpBase.td"

def TTL_DFBAccessOpInterface : OpInterface<"DFBAccessOpInterface"> {
  let description = [{
    Operations that expose dataflow buffer dependencies, ordered protocol
    actions, and typed non-transactional accesses. Dependencies identify every
    statically known DFB storage access. Protocol effects identify synchronous
    reserve, push, wait, and pop actions that occur on every operation
    execution, in execution order. Non-transactional summaries describe
    synchronous accesses that do not change queue state. Unknown access is
    distinct from a malformed or partial summary and conservatively applies to
    user-managed DFB storage.
  }];

  let cppNamespace = "::mlir::tt::ttl";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return DFB dependencies addressed by DFBProtocolEffect::dependencyIndex.",
      /*retTy=*/"::llvm::SmallVector<::mlir::Value>",
      /*methodName=*/"getDFBDependencyOperands",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/"Return ordered, synchronous protocol effects.",
      /*retTy=*/"::llvm::SmallVector<::mlir::tt::ttl::DFBProtocolEffect>",
      /*methodName=*/"getDFBProtocolEffects",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/"Return typed, synchronous non-transactional accesses.",
      /*retTy=*/"::llvm::SmallVector<::mlir::tt::ttl::DFBNonTransactionalAccess>",
      /*methodName=*/"getDFBNonTransactionalAccesses",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return {};">,
    InterfaceMethod<
      /*desc=*/"Return DFBs whose physical indices are consumed by the operation.",
      /*retTy=*/"::llvm::SmallVector<::mlir::Value>",
      /*methodName=*/"getDFBIndexOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return {};">,
    InterfaceMethod<
      /*desc=*/"Return true when the operation may access unlisted user DFBs.",
      /*retTy=*/"bool",
      /*methodName=*/"hasUnknownDFBAccess",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return false;">
  ];
}

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
    result evaluates to `true` on a per-PipeNet role-defined execution set.
    Predicates with graph records match both the launch node and logical-device
    endpoint; predicates without graph records match the launch node alone.
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
    InterfaceMethod<
      /*desc=*/"Return the graph record table, when the predicate is device-aware.",
      /*retTy=*/"::mlir::tt::ttl::PipeNetRecordsAttr",
      /*methodName=*/"getReferencedRecords",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->template getAttrOfType<
            ::mlir::tt::ttl::PipeNetRecordsAttr>("records");
      }]>,
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
      /*desc=*/"Return init operands for outputs whose initial mode requires "
               "one.",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getAccumulationInits",
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

def TTL_ComputePrimitiveOpInterface
    : OpInterface<"ComputePrimitiveOpInterface"> {
  let description = [{
    Operations that implement a target-independent compute primitive. The
    primitive identifies the hardware capability required by both tensor-level
    and tile-level forms without depending on operation-name dispatch.
  }];

  let cppNamespace = "::mlir::tt::ttl";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the target-independent compute primitive implemented "
               "by this operation.",
      /*retTy=*/"::mlir::tt::ttl::ComputePrimitive",
      /*methodName=*/"getComputePrimitive",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ConcreteOp::getComputePrimitiveKind();
      }]>
  ];
}

def TTL_DstAccessOpInterface : OpInterface<"DstAccessOpInterface"> {
  let description = [{
    Tile-level operations that expose concrete DST register reads, writes, and
    result residency. The interface is specific to DST behavior after tile
    lowering has assigned DST indices; it does not model dataflow buffer,
    packer, or generic memory effects.
  }];

  let cppNamespace = "::mlir::tt::ttl";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return DST slots read by this operation after tile execution semantics and DST residency are resolved.",
      /*retTy=*/"::mlir::FailureOr<::llvm::SmallVector<::mlir::tt::ttl::DstFootprint, 2>>",
      /*methodName=*/"getDstReadFootprints",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::tt::ttl::getDefaultDstReadFootprints($_op.getOperation());
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return DST slots written by this operation.",
      /*retTy=*/"::llvm::SmallVector<::mlir::tt::ttl::DstFootprint, 2>",
      /*methodName=*/"getDstWriteFootprints",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::tt::ttl::getDefaultDstWriteFootprints($_op.getOperation());
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the DST footprint represented by `result` when it is a DST-resident tile result of this operation.",
      /*retTy=*/"::mlir::FailureOr<::mlir::tt::ttl::DstFootprint>",
      /*methodName=*/"getResultDstFootprint",
      /*args=*/(ins "::mlir::Value":$result),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::tt::ttl::getDefaultResultDstFootprint(
            $_op.getOperation(), result);
      }]
    >
  ];
}

def TTL_TileExecutionOpInterface : OpInterface<"TileExecutionOpInterface"> {
  let description = [{
    Tile operations expose target-independent execution semantics through this
    interface. An execution strategy determines operand routes and DST
    behavior independently of target capabilities.
  }];

  let cppNamespace = "::mlir::tt::ttl";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return execution strategies structurally permitted by the current operands.",
      /*retTy=*/"::llvm::SmallVector<::mlir::tt::ttl::TileExecutionStrategy, 2>",
      /*methodName=*/"getLegalExecutionStrategies",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::tt::ttl::getDefaultLegalTileExecutionStrategies(
            $_op.getOperation());
      }]>,
    InterfaceMethod<
      /*desc=*/"Return execution semantics for the supplied strategy.",
      /*retTy=*/"::mlir::FailureOr<::mlir::tt::ttl::TileExecutionInfo>",
      /*methodName=*/"getTileExecutionInfo",
      /*args=*/(ins "::std::optional<::mlir::tt::ttl::TileExecutionStrategy>":$strategy),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::tt::ttl::getDefaultTileExecutionInfo(
            $_op.getOperation(), strategy);
      }]>
  ];
}

#endif // TTLANG_DIALECT_TTL_IR_TTLINTERFACES_TD
