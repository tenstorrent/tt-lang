// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/ComputeTarget.h"

#include "ttlang/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttlang/Dialect/TTCore/IR/Utils.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <array>

namespace mlir::tt::ttl {

namespace {

bool isBFPDataType(ttcore::DataType dataType) {
  using ttcore::DataType;
  switch (dataType) {
  case DataType::BFP_Float8:
  case DataType::BFP_BFloat8:
  case DataType::BFP_Float4:
  case DataType::BFP_BFloat4:
  case DataType::BFP_Float2:
  case DataType::BFP_BFloat2:
    return true;
  case DataType::Float32:
  case DataType::Float16:
  case DataType::BFloat16:
  case DataType::UInt32:
  case DataType::UInt16:
  case DataType::UInt8:
  case DataType::Int32:
  case DataType::Bool:
    return false;
  }
}

bool isSupportedIntegerDataType(ttcore::DataType dataType) {
  return dataType == ttcore::DataType::Int32 ||
         dataType == ttcore::DataType::UInt32 ||
         dataType == ttcore::DataType::UInt16;
}

bool isSupportedPassthroughDataType(ttcore::DataType dataType) {
  return dataType == ttcore::DataType::BFloat16 ||
         dataType == ttcore::DataType::Float16 ||
         dataType == ttcore::DataType::Float32 ||
         isSupportedIntegerDataType(dataType) || isBFPDataType(dataType);
}

bool isIntegerDataType(ttcore::DataType dataType) {
  return !ttcore::isFloat(dataType);
}

bool hasTileShape(ttcore::TileType tileType, int64_t height, int64_t width) {
  return tileType.getHeight() == height && tileType.getWidth() == width;
}

bool isElementwiseComputeShape(ttcore::TileType tileType) {
  return (tileType.getHeight() == 16 || tileType.getHeight() == 32) &&
         (tileType.getWidth() == 16 || tileType.getWidth() == 32);
}

bool isMatmulKernelShape(ttcore::TileType tileType) {
  // TT-Metal e908c313 labels short-height tiles host-loopback-only, but the
  // device matmul matrix confirms that its matmul LLKs support them.
  return isElementwiseComputeShape(tileType) || hasTileShape(tileType, 1, 32) ||
         hasTileShape(tileType, 2, 32) || hasTileShape(tileType, 4, 32) ||
         hasTileShape(tileType, 8, 32);
}

class WormholeBlackholeComputeTargetEnvironment final
    : public ComputeTargetEnvironment {
public:
  LogicalResult
  validateKernelTileType(bool containsMatmul, ttcore::TileType tileType,
                         std::string &failureReason) const final {
    failureReason.clear();
    bool supportedShape = containsMatmul ? isMatmulKernelShape(tileType)
                                         : isElementwiseComputeShape(tileType);
    if (!supportedShape) {
      llvm::raw_string_ostream diagnostic(failureReason);
      diagnostic << "tile shape " << tileType.getHeight() << "x"
                 << tileType.getWidth();
      if (containsMatmul) {
        diagnostic << " is not supported by compute kernels containing "
                      "matmul; supported shapes are 1x32, 2x32, 4x32, 8x32, "
                      "16x16, 16x32, 32x16, and 32x32";
      } else {
        diagnostic << " is not supported by the current compute LLKs; "
                      "supported shapes are 16x16, 16x32, 32x16, and 32x32";
      }
      return failure();
    }

    constexpr std::array<int64_t, 2> defaultTileShape =
        ttcore::TileType::getDefaultShape();
    if (isBFPDataType(tileType.getDataType()) &&
        (tileType.getHeight() != defaultTileShape[0] ||
         tileType.getWidth() != defaultTileShape[1])) {
      llvm::raw_string_ostream diagnostic(failureReason);
      diagnostic << "BFP compute tiles require " << defaultTileShape[0] << "x"
                 << defaultTileShape[1] << " dimensions, got "
                 << tileType.getHeight() << "x" << tileType.getWidth();
      return failure();
    }
    return success();
  }

  LogicalResult
  validatePrimitiveDataType(ComputePrimitive primitive,
                            ttcore::TileType tileType,
                            std::string &failureReason) const final {
    failureReason.clear();
    ttcore::DataType dataType = tileType.getDataType();
    if (!isIntegerDataType(dataType) ||
        primitive == ComputePrimitive::Typecast) {
      return success();
    }

    bool isIntegerArithmetic = primitive == ComputePrimitive::Add ||
                               primitive == ComputePrimitive::Subtract ||
                               primitive == ComputePrimitive::Multiply;
    bool isIntegerBroadcast = primitive == ComputePrimitive::Broadcast;
    if ((isIntegerArithmetic || isIntegerBroadcast) &&
        isSupportedIntegerDataType(dataType)) {
      return success();
    }

    llvm::raw_string_ostream diagnostic(failureReason);
    diagnostic << "integer tile type " << tileType;
    if (isIntegerArithmetic) {
      diagnostic << " is not supported; integer add, subtract, and multiply "
                    "support si32, u32, and u16 tiles";
    } else if (isIntegerBroadcast) {
      diagnostic << " is not supported; integer broadcast supports si32, "
                    "u32, and u16 tiles";
    } else {
      diagnostic << " is not supported by this compute primitive";
    }
    return failure();
  }

  LogicalResult
  validatePassthroughTileType(ttcore::TileType tileType,
                              std::string &failureReason) const final {
    failureReason.clear();
    if (!isSupportedPassthroughDataType(tileType.getDataType())) {
      llvm::raw_string_ostream diagnostic(failureReason);
      diagnostic << "tile type " << tileType
                 << " is not supported; passthrough supports bf16, f16, f32, "
                    "BFP, si32, u32, and u16 tiles";
      return failure();
    }
    constexpr std::array<int64_t, 2> defaultTileShape =
        ttcore::TileType::getDefaultShape();
    if (isBFPDataType(tileType.getDataType()) &&
        (tileType.getHeight() != defaultTileShape[0] ||
         tileType.getWidth() != defaultTileShape[1])) {
      llvm::raw_string_ostream diagnostic(failureReason);
      diagnostic << "BFP tiles require " << defaultTileShape[0] << "x"
                 << defaultTileShape[1] << " dimensions, got "
                 << tileType.getHeight() << "x" << tileType.getWidth();
      return failure();
    }
    return success();
  }

  LogicalResult
  validateMatmulTileTypes(ttcore::TileType lhsType, ttcore::TileType rhsType,
                          ttcore::TileType resultType, bool transposeRhs,
                          std::string &failureReason) const final {
    if (failed(verifyMatmulTileTypes(lhsType, rhsType, resultType, transposeRhs,
                                     failureReason))) {
      return failure();
    }

    for (ttcore::TileType tileType : {lhsType, rhsType, resultType}) {
      std::string tileFailureReason;
      if (failed(validateKernelTileType(/*containsMatmul=*/true, tileType,
                                        tileFailureReason))) {
        failureReason = std::move(tileFailureReason);
        return failure();
      }
    }

    if (hasTileShape(lhsType, 16, 16)) {
      failureReason =
          "matmul lhs tile dimensions 16x16 are not implemented by the "
          "current compute LLKs";
      return failure();
    }
    if (!hasTileShape(rhsType, 32, 32) && !hasTileShape(rhsType, 32, 16) &&
        !hasTileShape(rhsType, 16, 32)) {
      llvm::raw_string_ostream diagnostic(failureReason);
      diagnostic << "matmul rhs tile dimensions " << rhsType.getHeight()
                 << "x" << rhsType.getWidth()
                 << " are not implemented by the current compute LLKs; "
                    "supported rhs dimensions are 16x32, 32x16, and 32x32";
      return failure();
    }
    if (transposeRhs && hasTileShape(rhsType, 32, 16)) {
      failureReason =
          "matmul transpose_rhs is not implemented for 32x16 rhs tiles";
      return failure();
    }
    if (transposeRhs && hasTileShape(lhsType, 32, 32) &&
        hasTileShape(rhsType, 16, 32)) {
      failureReason = "matmul tile dimensions lhs 32x32 and rhs 16x32 do not "
                      "support transpose_rhs in the current compute LLKs";
      return failure();
    }
    return success();
  }

};

class IntersectionComputeTargetEnvironment final
    : public ComputeTargetEnvironment {
public:
  explicit IntersectionComputeTargetEnvironment(
      SmallVector<std::unique_ptr<ComputeTargetEnvironment>, 2> environments)
      : environments(std::move(environments)) {}

  LogicalResult validateKernelTileType(bool containsMatmul,
                                       ttcore::TileType tileType,
                                       std::string &failureReason) const final {
    for (const std::unique_ptr<ComputeTargetEnvironment> &environment :
         environments) {
      if (failed(environment->validateKernelTileType(containsMatmul, tileType,
                                                     failureReason))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult
  validatePrimitiveDataType(ComputePrimitive primitive,
                            ttcore::TileType tileType,
                            std::string &failureReason) const final {
    for (const std::unique_ptr<ComputeTargetEnvironment> &environment :
         environments) {
      if (failed(environment->validatePrimitiveDataType(primitive, tileType,
                                                        failureReason))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult
  validatePassthroughTileType(ttcore::TileType tileType,
                              std::string &failureReason) const final {
    for (const std::unique_ptr<ComputeTargetEnvironment> &environment :
         environments) {
      if (failed(environment->validatePassthroughTileType(tileType,
                                                          failureReason))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult
  validateMatmulTileTypes(ttcore::TileType lhsType, ttcore::TileType rhsType,
                          ttcore::TileType resultType, bool transposeRhs,
                          std::string &failureReason) const final {
    for (const std::unique_ptr<ComputeTargetEnvironment> &environment :
         environments) {
      if (failed(environment->validateMatmulTileTypes(
              lhsType, rhsType, resultType, transposeRhs, failureReason))) {
        return failure();
      }
    }
    return success();
  }

private:
  SmallVector<std::unique_ptr<ComputeTargetEnvironment>, 2> environments;
};

using ComputeTargetFactory = std::unique_ptr<ComputeTargetEnvironment> (*)();

struct ComputeTargetRegistration {
  ttcore::Arch arch;
  ComputeTargetFactory create;
};

std::unique_ptr<ComputeTargetEnvironment>
createWormholeBlackholeTargetEnvironment() {
  return std::make_unique<WormholeBlackholeComputeTargetEnvironment>();
}

constexpr std::array<ComputeTargetRegistration, 2> computeTargetRegistrations =
    {{{ttcore::Arch::WormholeB0, &createWormholeBlackholeTargetEnvironment},
      {ttcore::Arch::Blackhole, &createWormholeBlackholeTargetEnvironment}}};

FailureOr<std::unique_ptr<ComputeTargetEnvironment>>
createTargetEnvironment(ttcore::Arch arch, std::string &failureReason) {
  auto registration = llvm::find_if(
      computeTargetRegistrations, [&](const ComputeTargetRegistration &entry) {
        return entry.arch == arch;
      });
  if (registration != computeTargetRegistrations.end()) {
    return registration->create();
  }
  failureReason =
      "Quasar compute LLK capabilities are not implemented by TT-Lang";
  return failure();
}

std::unique_ptr<ComputeTargetEnvironment> createCommonTargetEnvironment() {
  SmallVector<std::unique_ptr<ComputeTargetEnvironment>, 2> environments;
  for (const ComputeTargetRegistration &registration :
       computeTargetRegistrations) {
    environments.push_back(registration.create());
  }
  return std::make_unique<IntersectionComputeTargetEnvironment>(
      std::move(environments));
}

FailureOr<std::optional<ttcore::Arch>>
getDeviceArch(ModuleOp module, std::string &failureReason) {
  auto systemDesc = module->getAttrOfType<ttcore::SystemDescAttr>(
      ttcore::SystemDescAttr::name);
  auto device =
      ttcore::lookupDeviceOp(module, ttcore::getDefaultDeviceName());
  if (!systemDesc || !device) {
    return std::optional<ttcore::Arch>();
  }

  ArrayRef<unsigned> chipIds = device.getDeviceAttr().getChipIds();
  if (chipIds.empty()) {
    failureReason = "default device has no selected chip";
    return failure();
  }
  auto invalidChip = llvm::find_if(chipIds, [&](unsigned chipId) {
    return chipId >= systemDesc.getChipDescIndices().size();
  });
  if (invalidChip != chipIds.end()) {
    failureReason = "default device selects chip " +
                    std::to_string(*invalidChip) +
                    " outside the system description";
    return failure();
  }
  ttcore::Arch arch =
      systemDesc.getChipDesc(chipIds.front()).getArch().getValue();
  if (llvm::any_of(llvm::drop_begin(chipIds), [&](unsigned chipId) {
        return systemDesc.getChipDesc(chipId).getArch().getValue() != arch;
      })) {
    failureReason = "default device selects chips with different architectures";
    return failure();
  }
  return std::optional<ttcore::Arch>(arch);
}

} // namespace

FailureOr<std::unique_ptr<ComputeTargetEnvironment>>
ComputeTargetEnvironment::get(Operation *operation,
                              std::string &failureReason) {
  failureReason.clear();
  ModuleOp module = dyn_cast<ModuleOp>(operation);
  if (!module) {
    module = operation->getParentOfType<ModuleOp>();
  }
  if (!module) {
    failureReason = "operation is not nested in a module";
    return failure();
  }

  std::optional<ttcore::Arch> attributeArch;
  Attribute rawTargetArch = module->getAttr(kTargetArchAttrName);
  auto targetArch = dyn_cast_or_null<StringAttr>(rawTargetArch);
  if (rawTargetArch && !targetArch) {
    failureReason = (kTargetArchAttrName + " must be a string attribute").str();
    return failure();
  }
  if (targetArch) {
    attributeArch = ttcore::symbolizeArch(targetArch.getValue());
    if (!attributeArch) {
      failureReason = (kTargetArchAttrName + " has unsupported value '" +
                       targetArch.getValue() + "'")
                          .str();
      return failure();
    }
  }

  FailureOr<std::optional<ttcore::Arch>> deviceArch =
      getDeviceArch(module, failureReason);
  if (failed(deviceArch)) {
    return failure();
  }
  if (attributeArch && *deviceArch && *attributeArch != **deviceArch) {
    failureReason =
        (kTargetArchAttrName + " does not match the selected device arch")
            .str();
    return failure();
  }
  std::optional<ttcore::Arch> arch = attributeArch ? attributeArch : *deviceArch;
  if (!arch) {
    return createCommonTargetEnvironment();
  }
  return createTargetEnvironment(*arch, failureReason);
}

LogicalResult ComputeTargetEnvironment::validateOperation(
    Operation *operation, bool containsMatmul,
    std::string &failureReason) const {
  std::optional<ComputePrimitive> primitive =
      getComputePrimitive(operation);
  if (!primitive) {
    failureReason = "has no compute-target capability classification";
    return failure();
  }

  if (*primitive == ComputePrimitive::Typecast) {
    FailureOr<ttcore::TileType> inputType =
        getTileType(operation->getOperand(0).getType());
    FailureOr<ttcore::TileType> resultType =
        getTileType(operation->getResult(0).getType());
    if (succeeded(inputType) && succeeded(resultType) &&
        failed(
            verifyTypecastTileTypes(*inputType, *resultType, failureReason))) {
      return failure();
    }
  }

  for (Type type : llvm::concat<Type>(operation->getOperandTypes(),
                                      operation->getResultTypes())) {
    FailureOr<ttcore::TileType> tileType = getTileType(type);
    if (failed(tileType)) {
      continue;
    }
    if (*primitive == ComputePrimitive::Passthrough) {
      if (failed(validatePassthroughTileType(*tileType, failureReason))) {
        return failure();
      }
      continue;
    }
    if (failed(validateKernelTileType(containsMatmul, *tileType,
                                      failureReason)) ||
        failed(validatePrimitiveDataType(*primitive, *tileType,
                                         failureReason))) {
      return failure();
    }
  }

  auto validateMatmul = [&](Type lhs, Type rhs, Type result,
                            bool transposeRhs) {
    FailureOr<ttcore::TileType> lhsType = getTileType(lhs);
    FailureOr<ttcore::TileType> rhsType = getTileType(rhs);
    FailureOr<ttcore::TileType> resultType = getTileType(result);
    if (failed(lhsType) || failed(rhsType) || failed(resultType)) {
      failureReason =
          "expected matmul operands and result to contain tile types";
      return failure();
    }
    return validateMatmulTileTypes(*lhsType, *rhsType, *resultType,
                                   transposeRhs, failureReason);
  };
  if (auto matmul = dyn_cast<MatmulOp>(operation)) {
    return validateMatmul(matmul.getLhs().getType(), matmul.getRhs().getType(),
                          matmul.getResult().getType(),
                          matmul.getTransposeRhs());
  }
  if (auto matmul = dyn_cast<TileMatmulBlockOp>(operation)) {
    return validateMatmul(matmul.getLhs().getType(), matmul.getRhs().getType(),
                          matmul.getResult().getType(),
                          matmul.getTransposeRhs());
  }
  return success();
}

std::optional<ComputePrimitive> getComputePrimitive(Operation *operation) {
  if (isa<AddOp, AddTileOp>(operation)) {
    return ComputePrimitive::Add;
  }
  if (isa<SubOp, SubTileOp>(operation)) {
    return ComputePrimitive::Subtract;
  }
  if (isa<MulOp, MulTileOp>(operation)) {
    return ComputePrimitive::Multiply;
  }
  if (isa<DivOp, DivTileOp, EqOp, EqTileOp, NeOp, NeTileOp, GtOp, GtTileOp,
          LtOp, LtTileOp, MaxOp, MaxTileOp, MinOp, MinTileOp>(operation)) {
    return ComputePrimitive::ElementwiseBinary;
  }
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  if (isa<TTL_OP##Op, TILE_OP>(operation))                                    \
    return ComputePrimitive::ElementwiseUnary;
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"
  if (isa<BlockBroadcastOp, TileBcastOp>(operation)) {
    return ComputePrimitive::Broadcast;
  }
  if (isa<ReduceOp, TileReduceOp>(operation)) {
    return ComputePrimitive::Reduce;
  }
  if (isa<TransposeOp, TileTransposeOp>(operation)) {
    return ComputePrimitive::Transpose;
  }
  if (isa<FillOp, TileFillOp>(operation)) {
    return ComputePrimitive::Fill;
  }
  if (isa<MatmulOp, TileMatmulBlockOp>(operation)) {
    return ComputePrimitive::Matmul;
  }
  if (isa<TypecastOp, TileTypecastOp>(operation)) {
    return ComputePrimitive::Typecast;
  }
  if (isa<MulUnaryConstOp, TileMulUnaryConstOp>(operation)) {
    return ComputePrimitive::MultiplyByConstant;
  }
  if (isa<StoreOp, TileStoreOp>(operation)) {
    return ComputePrimitive::Passthrough;
  }
  return std::nullopt;
}

bool containsMatmulOperation(Operation *scope) {
  WalkResult result = scope->walk([](Operation *operation) {
    return isa<MatmulOp, TileMatmulBlockOp>(operation)
               ? WalkResult::interrupt()
               : WalkResult::advance();
  });
  return result.wasInterrupted();
}

} // namespace mlir::tt::ttl
