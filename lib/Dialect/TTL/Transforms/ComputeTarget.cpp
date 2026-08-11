// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/ComputeTarget.h"

#include "ttlang/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
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

bool isStandardComputeShape(ttcore::TileType tileType) {
  return (tileType.getHeight() == 16 || tileType.getHeight() == 32) &&
         (tileType.getWidth() == 16 || tileType.getWidth() == 32);
}

bool isShortHeightComputeShape(ttcore::TileType tileType) {
  return (tileType.getHeight() == 1 || tileType.getHeight() == 2 ||
          tileType.getHeight() == 4 || tileType.getHeight() == 8) &&
         tileType.getWidth() == 32;
}

bool isSupportedShortHeightFloatDataType(ttcore::TileType tileType) {
  ttcore::DataType dataType = tileType.getDataType();
  return dataType == ttcore::DataType::BFloat16 ||
         dataType == ttcore::DataType::Float32;
}

bool isComputeShape(ttcore::TileType tileType) {
  return isStandardComputeShape(tileType) ||
         isShortHeightComputeShape(tileType);
}

bool supportsShortHeightTiles(ComputePrimitive primitive) {
  switch (primitive) {
  case ComputePrimitive::Add:
  case ComputePrimitive::Subtract:
  case ComputePrimitive::Multiply:
  case ComputePrimitive::ElementwiseBinary:
  case ComputePrimitive::ElementwiseUnary:
  case ComputePrimitive::Fill:
  case ComputePrimitive::Matmul:
  case ComputePrimitive::MultiplyByConstant:
  case ComputePrimitive::Broadcast:
  case ComputePrimitive::Reduce:
    return true;
  case ComputePrimitive::Transpose:
  case ComputePrimitive::Typecast:
    return false;
  case ComputePrimitive::Passthrough:
    llvm_unreachable("passthrough tile shapes are validated separately");
  }
}

class WormholeBlackholeComputeTargetEnvironment final
    : public ComputeTargetEnvironment {
public:
  LogicalResult validateKernelTileType(ttcore::TileType tileType,
                                       std::string &failureReason) const final {
    failureReason.clear();
    if (!isComputeShape(tileType)) {
      llvm::raw_string_ostream diagnostic(failureReason);
      diagnostic << "tile shape " << tileType.getHeight() << "x"
                 << tileType.getWidth()
                 << " is not supported by the current compute LLKs; "
                    "supported shapes are 1x32, 2x32, 4x32, 8x32, 16x16, "
                    "16x32, 32x16, and 32x32";
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
  validatePrimitiveTileShape(ComputePrimitive primitive,
                             ttcore::TileType tileType,
                             std::string &failureReason) const final {
    failureReason.clear();
    if (!isShortHeightComputeShape(tileType) ||
        supportsShortHeightTiles(primitive)) {
      return success();
    }

    llvm::raw_string_ostream diagnostic(failureReason);
    diagnostic << "tile shape " << tileType.getHeight() << "x"
               << tileType.getWidth()
               << " is not supported by this compute primitive; "
                  "short-height tiles are supported by elementwise, fill, "
                  "matmul, 8x32 row reduction, and 8x32 column broadcast "
                  "compute primitives";
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

  LogicalResult validateBroadcast(const BroadcastCapability &capability,
                                  std::string &failureReason) const final {
    failureReason.clear();
    if (!capability.tileBroadcast) {
      return success();
    }
    bool hasShortHeightTile = isShortHeightComputeShape(capability.inputType) ||
                              isShortHeightComputeShape(capability.resultType);
    if (!hasShortHeightTile) {
      return success();
    }

    if (!hasTileShape(capability.inputType, 8, 32) ||
        !hasTileShape(capability.resultType, 8, 32) ||
        capability.inputType != capability.resultType) {
      failureReason = "short-height broadcast supports only matching 8x32 "
                      "input and result tile types";
      return failure();
    }
    if (!isSupportedShortHeightFloatDataType(capability.inputType) ||
        !isSupportedShortHeightFloatDataType(capability.resultType)) {
      failureReason = "8x32 broadcast supports only bf16 and f32 tiles";
      return failure();
    }
    if (*capability.tileBroadcast != BcastType::Col) {
      failureReason = "8x32 broadcast supports only column broadcast";
      return failure();
    }
    return success();
  }

  LogicalResult validateReduction(const ReductionCapability &capability,
                                  std::string &failureReason) const final {
    failureReason.clear();
    bool hasShortHeightTile =
        isShortHeightComputeShape(capability.inputType) ||
        isShortHeightComputeShape(capability.scalerType) ||
        isShortHeightComputeShape(capability.resultType);
    if (!hasShortHeightTile) {
      return success();
    }

    if (!hasTileShape(capability.inputType, 8, 32) ||
        !hasTileShape(capability.scalerType, 8, 32) ||
        !hasTileShape(capability.resultType, 8, 32) ||
        capability.inputType != capability.scalerType ||
        capability.inputType != capability.resultType) {
      failureReason = "short-height reduction supports only matching 8x32 "
                      "input, scaler, and result tile types";
      return failure();
    }
    if (!isSupportedShortHeightFloatDataType(capability.inputType) ||
        !isSupportedShortHeightFloatDataType(capability.scalerType) ||
        !isSupportedShortHeightFloatDataType(capability.resultType)) {
      failureReason = "8x32 reduction supports only bf16 and f32 tiles";
      return failure();
    }
    if (capability.reduceDimension != ttkernel::ReduceDim::Row) {
      failureReason = "8x32 reduction supports only row reduction";
      return failure();
    }
    if (capability.reduceType != ReduceType::Sum &&
        capability.reduceType != ReduceType::Max) {
      failureReason = "8x32 reduction supports only sum and max";
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
      if (failed(validateKernelTileType(tileType, tileFailureReason))) {
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
      diagnostic << "matmul rhs tile dimensions " << rhsType.getHeight() << "x"
                 << rhsType.getWidth()
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

  LogicalResult validateKernelTileType(ttcore::TileType tileType,
                                       std::string &failureReason) const final {
    for (const std::unique_ptr<ComputeTargetEnvironment> &environment :
         environments) {
      if (failed(
              environment->validateKernelTileType(tileType, failureReason))) {
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
  validatePrimitiveTileShape(ComputePrimitive primitive,
                             ttcore::TileType tileType,
                             std::string &failureReason) const final {
    for (const std::unique_ptr<ComputeTargetEnvironment> &environment :
         environments) {
      if (failed(environment->validatePrimitiveTileShape(primitive, tileType,
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

  LogicalResult validateBroadcast(const BroadcastCapability &capability,
                                  std::string &failureReason) const final {
    for (const std::unique_ptr<ComputeTargetEnvironment> &environment :
         environments) {
      if (failed(environment->validateBroadcast(capability, failureReason))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult validateReduction(const ReductionCapability &capability,
                                  std::string &failureReason) const final {
    for (const std::unique_ptr<ComputeTargetEnvironment> &environment :
         environments) {
      if (failed(environment->validateReduction(capability, failureReason))) {
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

FailureOr<ttcore::TileType> getRequiredTileType(Type type, StringRef role,
                                                std::string &failureReason) {
  FailureOr<ttcore::TileType> tileType = getTileType(type);
  if (failed(tileType)) {
    llvm::raw_string_ostream diagnostic(failureReason);
    diagnostic << "expected " << role << " to contain a tile type";
    return failure();
  }
  return *tileType;
}

FailureOr<BroadcastCapability>
getBroadcastCapability(Operation *operation, std::string &failureReason) {
  if (auto broadcast = dyn_cast<BlockBroadcastOp>(operation)) {
    FailureOr<ttcore::TileType> inputType = getRequiredTileType(
        broadcast.getInput().getType(), "broadcast input", failureReason);
    if (failed(inputType)) {
      return failure();
    }
    FailureOr<ttcore::TileType> resultType = getRequiredTileType(
        broadcast.getResult().getType(), "broadcast result", failureReason);
    if (failed(resultType)) {
      return failure();
    }
    auto inputTensorType =
        dyn_cast<RankedTensorType>(broadcast.getInput().getType());
    if (!inputTensorType) {
      failureReason = "expected broadcast input to be a ranked tensor";
      return failure();
    }
    return BroadcastCapability{
        *inputType, *resultType,
        getTileBroadcastType(broadcast.getDims(), inputTensorType.getRank())};
  }
  if (auto broadcast = dyn_cast<TileBcastOp>(operation)) {
    FailureOr<ttcore::TileType> inputType = getRequiredTileType(
        broadcast.getInput().getType(), "broadcast input", failureReason);
    if (failed(inputType)) {
      return failure();
    }
    FailureOr<ttcore::TileType> resultType = getRequiredTileType(
        broadcast.getResult().getType(), "broadcast result", failureReason);
    if (failed(resultType)) {
      return failure();
    }
    return BroadcastCapability{*inputType, *resultType,
                               broadcast.getBcastType()};
  }
  failureReason = "has no broadcast capability representation";
  return failure();
}

FailureOr<ReductionCapability>
getReductionCapability(Operation *operation, std::string &failureReason) {
  auto buildCapability = [&](Type input, Type scaler, Type result,
                             ReduceType reduceType,
                             ttkernel::ReduceDim reduceDimension)
      -> FailureOr<ReductionCapability> {
    FailureOr<ttcore::TileType> inputType =
        getRequiredTileType(input, "reduction input", failureReason);
    if (failed(inputType)) {
      return failure();
    }
    FailureOr<ttcore::TileType> scalerType =
        getRequiredTileType(scaler, "reduction scaler", failureReason);
    if (failed(scalerType)) {
      return failure();
    }
    FailureOr<ttcore::TileType> resultType =
        getRequiredTileType(result, "reduction result", failureReason);
    if (failed(resultType)) {
      return failure();
    }
    return ReductionCapability{*inputType, *scalerType, *resultType, reduceType,
                               reduceDimension};
  };

  if (auto reduction = dyn_cast<ReduceOp>(operation)) {
    auto inputType = dyn_cast<RankedTensorType>(reduction.getInput().getType());
    if (!inputType) {
      failureReason = "expected reduction input to be a ranked tensor";
      return failure();
    }
    FailureOr<ttkernel::ReduceDim> reduceDimension =
        getReduceDimension(reduction.getDims(), inputType.getRank());
    if (failed(reduceDimension)) {
      failureReason = "has unsupported reduction dimensions";
      return failure();
    }
    return buildCapability(reduction.getInput().getType(),
                           reduction.getScaler().getType(),
                           reduction.getResult().getType(),
                           reduction.getReduceType(), *reduceDimension);
  }
  if (auto reduction = dyn_cast<TileReduceOp>(operation)) {
    return buildCapability(reduction.getInput().getType(),
                           reduction.getScaler().getType(),
                           reduction.getResult().getType(),
                           reduction.getReduceType(), reduction.getReduceDim());
  }
  failureReason = "has no reduction capability representation";
  return failure();
}

} // namespace

FailureOr<std::unique_ptr<ComputeTargetEnvironment>>
ComputeTargetEnvironment::get(Operation *operation,
                              std::string &failureReason) {
  FailureOr<std::optional<ttcore::Arch>> arch =
      resolveTargetArch(operation, failureReason);
  if (failed(arch)) {
    return failure();
  }
  if (!*arch) {
    return createCommonTargetEnvironment();
  }
  return createTargetEnvironment(**arch, failureReason);
}

LogicalResult
ComputeTargetEnvironment::validateOperation(Operation *operation,
                                            std::string &failureReason) const {
  std::optional<ComputePrimitive> primitive = getComputePrimitive(operation);
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
    if (failed(validateKernelTileType(*tileType, failureReason)) ||
        failed(
            validatePrimitiveTileShape(*primitive, *tileType, failureReason)) ||
        failed(
            validatePrimitiveDataType(*primitive, *tileType, failureReason))) {
      return failure();
    }
  }

  if (*primitive == ComputePrimitive::Broadcast) {
    FailureOr<BroadcastCapability> capability =
        getBroadcastCapability(operation, failureReason);
    return failed(capability) ? failure()
                              : validateBroadcast(*capability, failureReason);
  }
  if (*primitive == ComputePrimitive::Reduce) {
    FailureOr<ReductionCapability> capability =
        getReductionCapability(operation, failureReason);
    return failed(capability) ? failure()
                              : validateReduction(*capability, failureReason);
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
  auto primitiveOp = dyn_cast<ComputePrimitiveOpInterface>(operation);
  return primitiveOp ? std::optional(primitiveOp.getComputePrimitive())
                     : std::nullopt;
}

} // namespace mlir::tt::ttl
