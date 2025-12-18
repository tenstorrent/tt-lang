// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/BufferizableOpInterfaceImpl.h"

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
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

} // namespace

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension<TTLDialect>(+[](MLIRContext *ctx, TTLDialect *dialect) {
    (void)dialect;
    AttachCBOp::attachInterface<AttachCBOpInterface>(*ctx);
  });
}

} // namespace mlir::tt::ttl
