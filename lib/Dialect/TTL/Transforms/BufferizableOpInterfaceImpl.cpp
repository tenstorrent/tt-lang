// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/BufferizableOpInterfaceImpl.h"

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::tt::ttl {
namespace {

// Pull these into scope so the external model reads cleanly without repeatedly
// qualifying bufferization:: everywhere.
using bufferization::AliasingValueList;
using bufferization::AnalysisState;
using bufferization::BufferizableOpInterface;
using bufferization::BufferizationOptions;
using bufferization::BufferizationState;
using bufferization::BufferLikeType;
using bufferization::BufferRelation;

/// ttl.attach_cb is a pure aliasing op: it forwards its tensor/memref operand
/// to the result while preserving the CB metadata. Bufferization therefore:
///  - marks the operand as neither reading nor writing memory,
///  - reports operand/result equivalence,
///  - replaces the op with a new attach that carries the bufferized SSA value.
struct AttachCBOpInterface
    : public BufferizableOpInterface::ExternalModel<AttachCBOpInterface,
                                                    AttachCBOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &) const {
    if (opOperand.getOperandNumber() == 0) {
      return {{op->getResult(0), BufferRelation::Equivalent}};
    }
    return {};
  }

  BufferRelation bufferRelation(Operation *, OpOperand &,
                                const AnalysisState &) const {
    return BufferRelation::Equivalent;
  }

  bool isWritable(Operation *, Value, const AnalysisState &) const {
    // ttl.attach_cb is purely aliasing; writability follows the alias.
    return true;
  }

  FailureOr<BufferLikeType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    auto attachOp = cast<AttachCBOp>(op);
    if (value == attachOp.getResult()) {
      return bufferization::getBufferType(attachOp.getTensor(), options, state,
                                          invocationStack);
    }
    return bufferization::getBufferType(value, options, state, invocationStack);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto attachOp = cast<AttachCBOp>(op);
    FailureOr<Value> buffer = bufferization::getBuffer(
        rewriter, attachOp.getTensor(), options, state);
    if (failed(buffer)) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<AttachCBOp>(attachOp, (*buffer).getType(),
                                            *buffer, attachOp.getCb());
    return success();
  }
};

template <typename OpTy>
struct CBViewOpInterface
    : public BufferizableOpInterface::ExternalModel<CBViewOpInterface<OpTy>,
                                                    OpTy> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                      const AnalysisState &) const {
    return {};
  }

  BufferRelation bufferRelation(Operation *, OpOperand &,
                                const AnalysisState &) const {
    return BufferRelation::Unknown;
  }

  bool isWritable(Operation *, Value, const AnalysisState &) const {
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &) const {
    auto cbOp = cast<OpTy>(op);
    Type resultType = cbOp->getResult(0).getType();
    if (mlir::isa<BaseMemRefType>(resultType)) {
      return mlir::success();
    }

    auto tensorType = mlir::dyn_cast<RankedTensorType>(resultType);
    if (!tensorType) {
      return op->emitError()
             << "expected ranked tensor result prior to bufferization";
    }

    FailureOr<BaseMemRefType> bufferType =
        bufferization::getMemRefType(tensorType, options);
    if (failed(bufferType)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<OpTy>(cbOp, *bufferType, cbOp.getCb());
    return success();
  }
};

struct CopyOpInterface
    : public BufferizableOpInterface::ExternalModel<CopyOpInterface, CopyOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &) const {
    auto copy = cast<CopyOp>(op);
    if (opOperand.getOperandNumber() != 0) {
      return false;
    }
    return !copy.isSrcCircularBuffer();
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &) const {
    auto copy = cast<CopyOp>(op);
    if (opOperand.getOperandNumber() != 1) {
      return false;
    }
    return copy.isSrcCircularBuffer();
  }

  AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                      const AnalysisState &) const {
    return {};
  }

  BufferRelation bufferRelation(Operation *, OpOperand &,
                                const AnalysisState &) const {
    return BufferRelation::Unknown;
  }

  bool isWritable(Operation *, Value, const AnalysisState &) const {
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto copy = cast<CopyOp>(op);
    Value newSrc = copy.getSrc();
    Value newDst = copy.getDst();

    auto convertIfTensor = [&](Value oldVal,
                               Value &newVal) -> FailureOr<RankedTensorType> {
      auto tensorTy = mlir::dyn_cast<RankedTensorType>(oldVal.getType());
      if (!tensorTy) {
        return RankedTensorType();
      }
      FailureOr<Value> buffer =
          bufferization::getBuffer(rewriter, oldVal, options, state);
      if (failed(buffer)) {
        return failure();
      }
      newVal = *buffer;
      return tensorTy;
    };

    FailureOr<RankedTensorType> newTensorTy;
    if (!copy.isSrcCircularBuffer()) {
      newTensorTy = convertIfTensor(copy.getSrc(), newSrc);
      if (failed(newTensorTy)) {
        return failure();
      }
    }
    if (!copy.isDstCircularBuffer()) {
      auto converted = convertIfTensor(copy.getDst(), newDst);
      if (failed(converted)) {
        return failure();
      }
      if (*converted) {
        newTensorTy = converted;
      }
    }

    TypeAttr tensorAttr;
    if (*newTensorTy) {
      tensorAttr = TypeAttr::get(*newTensorTy);
    } else {
      tensorAttr = copy.getTensorTypeAttr();
    }
    rewriter.replaceOpWithNewOp<CopyOp>(copy, copy.getResult().getType(),
                                        newSrc, newDst, tensorAttr);

    return success();
  }
};

} // namespace

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension<TTLDialect>(+[](MLIRContext *ctx, TTLDialect *dialect) {
    (void)dialect;
    AttachCBOp::attachInterface<AttachCBOpInterface>(*ctx);
    CBReserveOp::attachInterface<CBViewOpInterface<CBReserveOp>>(*ctx);
    CBWaitOp::attachInterface<CBViewOpInterface<CBWaitOp>>(*ctx);
    CopyOp::attachInterface<CopyOpInterface>(*ctx);
  });
}

} // namespace mlir::tt::ttl
