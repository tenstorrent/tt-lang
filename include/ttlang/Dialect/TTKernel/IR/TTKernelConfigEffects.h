// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTKERNEL_IR_TTKERNELCONFIGEFFECTS_H
#define TTLANG_DIALECT_TTKERNEL_IR_TTKERNELCONFIGEFFECTS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::tt::ttkernel {

/// MATH configuration selected by the most recent per-op init. It is hardware
/// state rather than memory, so it is disjoint from the default resource.
struct MathInitResource : public SideEffects::Resource::Base<MathInitResource> {
  StringRef getName() const final { return "ttkernel.math_init"; }
  SideEffects::Resource *getParent() const final { return nullptr; }
  bool isAddressable() const final { return false; }
};

/// Identity of a per-op init: its name, key operands, and inherent attributes.
/// Two inits with equal descriptors configure MATH identically.
struct MathInitDescriptor {
  StringAttr init;
  SmallVector<Value, 4> operands;
  DictionaryAttr attributes;

  bool operator==(const MathInitDescriptor &other) const {
    return init == other.init && operands == other.operands &&
           attributes == other.attributes;
  }
  bool operator!=(const MathInitDescriptor &other) const {
    return !(*this == other);
  }
};

/// MathInit access of one operation. A write without a descriptor leaves the
/// configuration unknown. A read always carries the required descriptor.
struct MathInitEffect {
  std::optional<MathInitDescriptor> descriptor;
};

/// Read and write of MathInit by one operation. Either may be absent. Calls
/// without declared effects write an unknown configuration. Nested regions
/// are not inspected.
struct MathInitEffects {
  std::optional<MathInitEffect> read;
  std::optional<MathInitEffect> write;
};

/// Returns the MathInit read and write of `op`. MathInit changes only through
/// declared effects, except that calls without declared effects reset it.
MathInitEffects getMathInitEffects(Operation *op);

/// Appends the effects of a TTKernel compute or init operation: conservative
/// reads and writes of memory plus its declared MathInit access.
void getHardwareConfigEffects(
    Operation *op, SmallVectorImpl<MemoryEffects::EffectInstance> &effects);

/// Implements MemoryEffectOpInterface::getEffects with
/// getHardwareConfigEffects.
template <typename ConcreteType>
class TTKernelHardwareConfigEffectsTrait
    : public OpTrait::TraitBase<ConcreteType,
                                TTKernelHardwareConfigEffectsTrait> {
public:
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    getHardwareConfigEffects(this->getOperation(), effects);
  }
};

} // namespace mlir::tt::ttkernel

#endif
