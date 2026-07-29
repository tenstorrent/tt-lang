// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTKERNEL_IR_TTKERNELTRAITS_H
#define TTLANG_DIALECT_TTKERNEL_IR_TTKERNELTRAITS_H

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::tt::ttkernel {

template <typename ConcreteType>
class TTKernelFPUOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelFPUOpTrait> {};

template <typename ConcreteType>
class TTKernelSFPUOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelSFPUOpTrait> {};

template <typename ConcreteType>
class TTKernelInitOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelInitOpTrait> {};

template <typename ConcreteType>
class TTKernelUnaryOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelUnaryOpTrait> {
public:
  static constexpr int arity = 1;
};

template <typename ConcreteType>
class TTKernelBinaryOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelBinaryOpTrait> {
public:
  static constexpr int arity = 2;
};

template <typename ConcreteType>
class TTKernelTernaryOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelTernaryOpTrait> {
public:
  static constexpr int arity = 3;
};

template <typename ConcreteType>
class TTKernelDeviceZoneOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelDeviceZoneOpTrait> {
};

template <typename ConcreteType>
class TTKernelLayoutOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelLayoutOpTrait> {};

template <typename ConcreteType>
class TTKernelTridNocOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelTridNocOpTrait> {
public:
  static constexpr int32_t kMaxTrid = 15;
  static constexpr int32_t kNumNocs = 2;

  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    auto concreteOp = mlir::cast<ConcreteType>(op);

    auto tridValue = getConstantIntValue(concreteOp.getTrid());
    if (tridValue && (*tridValue < 0 || *tridValue > kMaxTrid)) {
      return op->emitOpError() << "trid must be in [0, " << kMaxTrid << "].";
    }

    mlir::Value noc = concreteOp.getNoc();
    if (noc) {
      auto nocValue = getConstantIntValue(noc);
      if (nocValue && (*nocValue < 0 || *nocValue >= kNumNocs)) {
        return op->emitOpError()
               << "noc must be in [0, " << (kNumNocs - 1) << "].";
      }
    }

    return mlir::success();
  }
};

/// Identifies operations that access NoC hardware.
template <typename ConcreteType>
class TTKernelNocOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelNocOpTrait> {};

/// Identifies operations that access the resident NoC read command.
template <typename ConcreteType>
class TTKernelNocReadCommandOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      TTKernelNocReadCommandOpTrait> {};

/// Identifies operations that access the resident NoC write command.
template <typename ConcreteType>
class TTKernelNocWriteCommandOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      TTKernelNocWriteCommandOpTrait> {};

/// Identifies operations that access the resident NoC atomic command.
template <typename ConcreteType>
class TTKernelNocAtomicCommandOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      TTKernelNocAtomicCommandOpTrait> {};

/// Identifies NoC operations that preserve resident command configuration.
template <typename ConcreteType>
class TTKernelNocCommandStatePreservingOpTrait
    : public mlir::OpTrait::TraitBase<
          ConcreteType, TTKernelNocCommandStatePreservingOpTrait> {};

/// NoC command resources tracked by stateful command optimizations.
enum class NocCommandClass {
  Read,
  Write,
  Atomic,
};

/// Return whether `op` may reprogram the selected resident NoC command.
///
/// An unclassified NoC operation conservatively reprograms every command
/// class. This prevents a newly added NoC operation from silently invalidating
/// stateful command reuse.
inline bool mayReprogramNocCommand(Operation *op,
                                   NocCommandClass commandClass) {
  if (!op->hasTrait<TTKernelNocOpTrait>() ||
      op->hasTrait<TTKernelNocCommandStatePreservingOpTrait>()) {
    return false;
  }

  bool accessesReadCommand = op->hasTrait<TTKernelNocReadCommandOpTrait>();
  bool accessesWriteCommand = op->hasTrait<TTKernelNocWriteCommandOpTrait>();
  bool accessesAtomicCommand = op->hasTrait<TTKernelNocAtomicCommandOpTrait>();
  if (!accessesReadCommand && !accessesWriteCommand && !accessesAtomicCommand) {
    return true;
  }

  switch (commandClass) {
  case NocCommandClass::Read:
    return accessesReadCommand;
  case NocCommandClass::Write:
    return accessesWriteCommand;
  case NocCommandClass::Atomic:
    return accessesAtomicCommand;
  }
  llvm_unreachable("unknown NoC command class");
}

} // namespace mlir::tt::ttkernel

#endif
