// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTKernel/IR/TTKernelConfigEffects.h"

#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"

#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::ttkernel {

namespace {

struct OperandSpan {
  const int32_t *data = nullptr;
  unsigned size = 0;
};

constexpr int32_t kOp0[] = {0};
constexpr int32_t kOp01[] = {0, 1};
constexpr int32_t kOpMatmulCompute[] = {0, 1, 5, 6, 7, 8};
constexpr int32_t kOpMatmulInit[] = {0, 1, 2, 3, 4, 5};

constexpr OperandSpan none = {};
constexpr OperandSpan c0 = {kOp0, 1};
constexpr OperandSpan i0 = {kOp0, 1};
constexpr OperandSpan c01 = {kOp01, 2};
constexpr OperandSpan i01 = {kOp01, 2};
constexpr OperandSpan cMatmul = {kOpMatmulCompute, 6};
constexpr OperandSpan iMatmul = {kOpMatmulInit, 6};

constexpr llvm::StringLiteral kBcastAttr[] = {"bcast_type"};
constexpr llvm::StringLiteral kReduceAttrs[] = {"reduce_type", "reduce_dim"};
constexpr llvm::StringLiteral kReuseAttrs[] = {"eltwise_binary_type",
                                               "reuse_type"};
constexpr llvm::StringLiteral kTypecastAttrs[] = {"in_dtype", "out_dtype"};
constexpr llvm::StringLiteral kDtypeAttr[] = {"dtype"};
constexpr llvm::StringLiteral kExpAttrs[] = {"approx", "scale",
                                             "input_clamping"};
constexpr llvm::StringLiteral kSfpuReduceAttrs[] = {"reduce_type",
                                                    "data_format"};

struct MathInitBinding {
  StringRef init;
  SmallVector<int32_t, 6> computeOperands;
  SmallVector<int32_t, 6> initOperands;
  SmallVector<StringRef, 4> attributes;
  bool canonicalizeExp = false;
};

struct InitSide {
  SmallVector<int32_t, 6> operands;
  SmallVector<StringRef, 4> attributes;
  bool canonicalizeExp = false;
};

SmallVector<int32_t, 6> operandVector(OperandSpan span) {
  if (span.size == 0) {
    return {};
  }
  return SmallVector<int32_t, 6>(span.data, span.data + span.size);
}

SmallVector<StringRef, 4> noAttrs() { return {}; }

template <size_t N>
SmallVector<StringRef, 4> attrList(const llvm::StringLiteral (&names)[N]) {
  return SmallVector<StringRef, 4>(std::begin(names), std::end(names));
}

SmallVector<StringRef, 4> bcastAttrs() { return attrList(kBcastAttr); }
SmallVector<StringRef, 4> reduceAttrs() { return attrList(kReduceAttrs); }
SmallVector<StringRef, 4> reuseAttrs() { return attrList(kReuseAttrs); }
SmallVector<StringRef, 4> typecastAttrs() { return attrList(kTypecastAttrs); }
SmallVector<StringRef, 4> dtypeAttrs() { return attrList(kDtypeAttr); }
SmallVector<StringRef, 4> expAttrs() { return attrList(kExpAttrs); }
SmallVector<StringRef, 4> sfpuReduceAttrs() {
  return attrList(kSfpuReduceAttrs);
}

MathInitBinding makeBinding(StringRef init, OperandSpan computeOperands,
                            OperandSpan initOperands,
                            SmallVector<StringRef, 4> attributes, bool exp) {
  MathInitBinding binding;
  binding.init = init;
  binding.computeOperands = operandVector(computeOperands);
  binding.initOperands = operandVector(initOperands);
  binding.attributes = std::move(attributes);
  binding.canonicalizeExp = exp;
  return binding;
}

std::optional<MathInitBinding> lookupComputeBinding(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<MathInitBinding>>(op)
#define MATH_INIT_AUTO(COMPUTE, INIT, COPS, IOPS, ATTRS, EXP)                  \
  .Case<COMPUTE>([](auto) {                                                    \
    return makeBinding(INIT::getOperationName(), COPS, IOPS, ATTRS, EXP);      \
  })
#define MATH_INIT_MANUAL(COMPUTE, INIT, COPS, IOPS, ATTRS, EXP)                \
  MATH_INIT_AUTO(COMPUTE, INIT, COPS, IOPS, ATTRS, EXP)
#include "ttlang/Dialect/TTKernel/IR/TTKernelMathInits.def"
      .Default([](Operation *) { return std::nullopt; });
}

const llvm::StringMap<InitSide> &initSides() {
  static const auto map = [] {
    llvm::StringMap<InitSide> sides;
    auto add = [&](StringRef name, InitSide side) {
      auto it = sides.find(name);
      if (it == sides.end()) {
        sides.insert({name, std::move(side)});
        return;
      }
      assert(it->second.operands == side.operands &&
             it->second.attributes == side.attributes &&
             it->second.canonicalizeExp == side.canonicalizeExp &&
             "per-op init key differs between compute operations");
    };
#define MATH_INIT_AUTO(COMPUTE, INIT, COPS, IOPS, ATTRS, EXP)                  \
  add(INIT::getOperationName(), InitSide{operandVector(IOPS), ATTRS, EXP});
#define MATH_INIT_MANUAL(COMPUTE, INIT, COPS, IOPS, ATTRS, EXP)                \
  MATH_INIT_AUTO(COMPUTE, INIT, COPS, IOPS, ATTRS, EXP)
#include "ttlang/Dialect/TTKernel/IR/TTKernelMathInits.def"
    return sides;
  }();
  return map;
}

bool isExcludedCompute(Operation *op) {
  return isa<MatmulTilesOp, RandTileOp>(op);
}

// The LLK programs MATH internally, or the body is an opaque SFPI region.
// Either way the configuration afterward is unknown.
bool writesUnknownMath(Operation *op) {
  return isa<ExperimentalRowNormalizationBlockOp, InvokeSFPIOp>(op);
}

bool isComputeOp(Operation *op) {
  return op->hasTrait<TTKernelFPUOpTrait>() ||
         op->hasTrait<TTKernelSFPUOpTrait>() ||
         isa<CopyTileOp, TransposeTileOp>(op);
}

Attribute defaultExpAttribute(MLIRContext *context, StringRef name) {
  if (name == "approx") {
    return BoolAttr::get(context, false);
  }
  if (name == "scale") {
    return IntegerAttr::get(IntegerType::get(context, 32), 0x3F800000);
  }
  if (name == "input_clamping") {
    return InputClampingAttr::get(context, InputClamping::ClampToNegative);
  }
  return {};
}

DictionaryAttr collectAttributes(Operation *op, ArrayRef<StringRef> names,
                                 bool canonicalizeExp) {
  MLIRContext *context = op->getContext();
  NamedAttrList attributes;
  for (StringRef name : names) {
    Attribute value;
    if (std::optional<Attribute> inherent = op->getInherentAttr(name)) {
      value = *inherent;
    }
    if ((!value) && canonicalizeExp) {
      value = defaultExpAttribute(context, name);
    }
    if (value) {
      attributes.append(StringAttr::get(context, name), value);
    }
  }
  return attributes.getDictionary(context);
}

/// Encodes a descriptor as effect parameters. Operands are stored as indices
/// into the operation that declares the effect.
Attribute encodeDescriptor(MLIRContext *context, StringRef init,
                           ArrayRef<int32_t> operands,
                           DictionaryAttr attributes) {
  return ArrayAttr::get(context, {StringAttr::get(context, init),
                                  DenseI32ArrayAttr::get(context, operands),
                                  attributes});
}

std::optional<MathInitDescriptor> decodeDescriptor(Operation *op,
                                                   Attribute params) {
  auto array = dyn_cast_if_present<ArrayAttr>(params);
  if (!array || array.size() != 3) {
    return std::nullopt;
  }
  auto init = dyn_cast<StringAttr>(array[0]);
  auto operands = dyn_cast<DenseI32ArrayAttr>(array[1]);
  auto attributes = dyn_cast<DictionaryAttr>(array[2]);
  if (!init || !operands || !attributes) {
    return std::nullopt;
  }
  MathInitDescriptor descriptor{init, {}, attributes};
  for (int32_t index : operands.asArrayRef()) {
    if (index < 0 || static_cast<unsigned>(index) >= op->getNumOperands()) {
      return std::nullopt;
    }
    descriptor.operands.push_back(op->getOperand(index));
  }
  return descriptor;
}

} // namespace

void getHardwareConfigEffects(
    Operation *op, SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());

  MLIRContext *context = op->getContext();
  if (std::optional<MathInitBinding> required = lookupComputeBinding(op)) {
    effects.emplace_back(
        MemoryEffects::Read::get(),
        encodeDescriptor(context, required->init, required->computeOperands,
                         collectAttributes(op, required->attributes,
                                           required->canonicalizeExp)),
        MathInitResource::get());
    return;
  }
  if (writesUnknownMath(op)) {
    effects.emplace_back(MemoryEffects::Write::get(), MathInitResource::get());
    return;
  }
  if (isComputeOp(op) && !isExcludedCompute(op)) {
    llvm_unreachable("compute operation has no MathInit binding");
  }

  if (!op->hasTrait<TTKernelInitOpTrait>() && !isa<TransposeInitOp>(op)) {
    return;
  }
  if (auto side = initSides().find(op->getName().getStringRef());
      side != initSides().end()) {
    effects.emplace_back(
        MemoryEffects::Write::get(),
        encodeDescriptor(context, op->getName().getStringRef(),
                         side->second.operands,
                         collectAttributes(op, side->second.attributes,
                                           side->second.canonicalizeExp)),
        MathInitResource::get());
    return;
  }
  effects.emplace_back(MemoryEffects::Write::get(), MathInitResource::get());
}

MathInitEffects getMathInitEffects(Operation *op) {
  auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectInterface) {
    if (isa<CallOpInterface>(op)) {
      MathInitEffects result;
      result.write = MathInitEffect{};
      return result;
    }
    return {};
  }

  SmallVector<MemoryEffects::EffectInstance, 2> effects;
  effectInterface.getEffectsOnResource(MathInitResource::get(), effects);
  MathInitEffects result;
  for (const MemoryEffects::EffectInstance &effect : effects) {
    MathInitEffect decoded{decodeDescriptor(op, effect.getParameters())};
    if (isa<MemoryEffects::Write>(effect.getEffect())) {
      if (!result.write) {
        result.write = decoded;
      }
    } else if (!result.read) {
      result.read = decoded;
    }
  }
  return result;
}

} // namespace mlir::tt::ttkernel
