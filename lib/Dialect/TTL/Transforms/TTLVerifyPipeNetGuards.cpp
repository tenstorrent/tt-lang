// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Verify PipeNet Guards Pass
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <tuple>

#define DEBUG_TYPE "ttl-verify-pipenet-guards"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLVERIFYPIPENETGUARDS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kKernelThreadAttrName = "ttl.kernel_thread";
constexpr llvm::StringLiteral kLaunchGridAttrName = "ttl.launch_grid";
constexpr llvm::StringLiteral kPipeNetIdsAttrName = "ttl.pipe_net_ids";
constexpr llvm::StringLiteral kPipeNetRolesAttrName = "ttl.pipe_net_roles";

enum class PipeRole : int64_t { Source = 0, Destination = 1 };

struct Coord {
  int64_t x = 0;
  int64_t y = 0;

  bool operator<(const Coord &rhs) const {
    return std::tie(x, y) < std::tie(rhs.x, rhs.y);
  }
};

struct Domain {
  bool known = true;
  std::set<Coord> nodes;

  static Domain unknown() { return {/*known=*/false, {}}; }

  bool isSubsetOf(const Domain &rhs) const {
    if (!known || !rhs.known) {
      return false;
    }
    for (Coord coord : nodes) {
      if (rhs.nodes.find(coord) == rhs.nodes.end()) {
        return false;
      }
    }
    return true;
  }
};

Domain domainUnion(const Domain &lhs, const Domain &rhs) {
  if (!lhs.known || !rhs.known) {
    return Domain::unknown();
  }
  Domain result;
  result.nodes = lhs.nodes;
  result.nodes.insert(rhs.nodes.begin(), rhs.nodes.end());
  return result;
}

Domain domainIntersect(const Domain &lhs, const Domain &rhs) {
  if (!lhs.known || !rhs.known) {
    return Domain::unknown();
  }
  Domain result;
  for (Coord coord : lhs.nodes) {
    if (rhs.nodes.find(coord) != rhs.nodes.end()) {
      result.nodes.insert(coord);
    }
  }
  return result;
}

Domain domainSubtract(const Domain &lhs, const Domain &rhs) {
  if (!lhs.known || !rhs.known) {
    return Domain::unknown();
  }
  Domain result;
  for (Coord coord : lhs.nodes) {
    if (rhs.nodes.find(coord) == rhs.nodes.end()) {
      result.nodes.insert(coord);
    }
  }
  return result;
}

Domain fullGridDomain(int64_t gridX, int64_t gridY) {
  Domain result;
  for (int64_t x = 0; x < gridX; ++x) {
    for (int64_t y = 0; y < gridY; ++y) {
      result.nodes.insert({x, y});
    }
  }
  return result;
}

Domain pipeSourceDomain(PipeType pipeType) {
  Domain result;
  result.nodes.insert({pipeType.getSrcX(), pipeType.getSrcY()});
  return result;
}

Domain pipeDestinationDomain(PipeType pipeType) {
  Domain result;
  for (int64_t x = pipeType.getDstStartX(); x <= pipeType.getDstEndX(); ++x) {
    for (int64_t y = pipeType.getDstStartY(); y <= pipeType.getDstEndY(); ++y) {
      result.nodes.insert({x, y});
    }
  }
  return result;
}

std::optional<int64_t> getIntegerAttrValue(Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    return intAttr.getInt();
  }
  return std::nullopt;
}

bool readI64Array(Operation *op, llvm::StringLiteral name,
                  SmallVectorImpl<int64_t> &values) {
  if (auto dense = op->getAttrOfType<DenseI64ArrayAttr>(name)) {
    values.append(dense.asArrayRef().begin(), dense.asArrayRef().end());
    return true;
  }
  auto array = op->getAttrOfType<ArrayAttr>(name);
  if (!array) {
    return false;
  }
  for (Attribute attr : array) {
    std::optional<int64_t> value = getIntegerAttrValue(attr);
    if (!value) {
      return false;
    }
    values.push_back(*value);
  }
  return true;
}

std::optional<std::pair<int64_t, int64_t>> readLaunchGrid(ModuleOp module) {
  SmallVector<int64_t> values;
  if (!readI64Array(module.getOperation(), kLaunchGridAttrName, values) ||
      values.size() != 2) {
    return std::nullopt;
  }
  if (values[0] <= 0 || values[1] <= 0) {
    return std::nullopt;
  }
  return std::pair<int64_t, int64_t>{values[0], values[1]};
}

std::optional<int64_t> getCBIndex(Value cb) {
  if (auto bindOp = cb.getDefiningOp<BindCBOp>()) {
    return bindOp.getCbIndex().getSExtValue();
  }
  if (auto castOp = cb.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (castOp.getInputs().size() == 1) {
      return getCBIndex(castOp.getInputs()[0]);
    }
  }
  return std::nullopt;
}

class GuardVerifier {
public:
  explicit GuardVerifier(ModuleOp module) : module(module) {}

  LogicalResult run() {
    collectPipeDomains();
    if (!hasPipes) {
      return success();
    }

    baseDomain = buildBaseDomain();

    SmallVector<func::FuncOp> kernels;
    module.walk([&](func::FuncOp func) {
      if (func->hasAttr(kKernelThreadAttrName)) {
        kernels.push_back(func);
      }
    });

    for (func::FuncOp func : kernels) {
      if (failed(walkRegion(func.getBody(), baseDomain))) {
        return failure();
      }
    }

    if (failed(verifyCBWaits())) {
      return failure();
    }

    inlinePipeNetScopes();
    return success();
  }

private:
  ModuleOp module;
  bool hasPipes = false;
  int64_t fallbackGridX = 1;
  int64_t fallbackGridY = 1;
  Domain baseDomain;
  std::map<int64_t, Domain> netSourceDomains;
  std::map<int64_t, Domain> netDestinationDomains;
  std::map<int64_t, Domain> cbProducerDomains;

  struct WaitUse {
    CBWaitOp op;
    Domain domain;
    int64_t cbIndex;
  };
  SmallVector<WaitUse> waitUses;

  void collectPipeDomains() {
    module.walk([&](CreatePipeOp pipe) {
      hasPipes = true;
      PipeType pipeType = cast<PipeType>(pipe.getResult().getType());
      int64_t pipeNetId = pipeType.getPipeNetId();
      netSourceDomains[pipeNetId] =
          domainUnion(netSourceDomains[pipeNetId], pipeSourceDomain(pipeType));
      netDestinationDomains[pipeNetId] = domainUnion(
          netDestinationDomains[pipeNetId], pipeDestinationDomain(pipeType));

      fallbackGridX = std::max(
          {fallbackGridX, pipeType.getSrcX() + 1, pipeType.getDstEndX() + 1});
      fallbackGridY = std::max(
          {fallbackGridY, pipeType.getSrcY() + 1, pipeType.getDstEndY() + 1});
    });
  }

  Domain buildBaseDomain() {
    if (auto launchGrid = readLaunchGrid(module)) {
      return fullGridDomain(launchGrid->first, launchGrid->second);
    }
    return fullGridDomain(fallbackGridX, fallbackGridY);
  }

  Domain getNetRoleDomain(int64_t pipeNetId, PipeRole role) {
    if (role == PipeRole::Source) {
      return netSourceDomains[pipeNetId];
    }
    return netDestinationDomains[pipeNetId];
  }

  std::optional<int64_t> evalIndex(Value value, Coord coord) {
    if (value.getDefiningOp<CoreXOp>()) {
      return coord.x;
    }
    if (value.getDefiningOp<CoreYOp>()) {
      return coord.y;
    }
    if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>()) {
      return constant.value();
    }
    if (auto constant = value.getDefiningOp<arith::ConstantIntOp>()) {
      return constant.value();
    }
    if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constant.getValue())) {
        return intAttr.getInt();
      }
    }
    if (auto castOp = value.getDefiningOp<arith::IndexCastOp>()) {
      return evalIndex(castOp.getIn(), coord);
    }
    if (auto addOp = value.getDefiningOp<arith::AddIOp>()) {
      auto lhs = evalIndex(addOp.getLhs(), coord);
      auto rhs = evalIndex(addOp.getRhs(), coord);
      if (lhs && rhs) {
        return *lhs + *rhs;
      }
    }
    if (auto subOp = value.getDefiningOp<arith::SubIOp>()) {
      auto lhs = evalIndex(subOp.getLhs(), coord);
      auto rhs = evalIndex(subOp.getRhs(), coord);
      if (lhs && rhs) {
        return *lhs - *rhs;
      }
    }
    if (auto mulOp = value.getDefiningOp<arith::MulIOp>()) {
      auto lhs = evalIndex(mulOp.getLhs(), coord);
      auto rhs = evalIndex(mulOp.getRhs(), coord);
      if (lhs && rhs) {
        return *lhs * *rhs;
      }
    }
    return std::nullopt;
  }

  std::optional<bool> evalBool(Value value, Coord coord) {
    if (auto constant = value.getDefiningOp<arith::ConstantIntOp>()) {
      if (constant.getType().isInteger(1)) {
        return constant.value() != 0;
      }
    }
    if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constant.getValue());
          intAttr && intAttr.getType().isInteger(1)) {
        return intAttr.getInt() != 0;
      }
    }
    if (auto cmpOp = value.getDefiningOp<arith::CmpIOp>()) {
      auto lhs = evalIndex(cmpOp.getLhs(), coord);
      auto rhs = evalIndex(cmpOp.getRhs(), coord);
      if (!lhs || !rhs) {
        return std::nullopt;
      }
      switch (cmpOp.getPredicate()) {
      case arith::CmpIPredicate::eq:
        return *lhs == *rhs;
      case arith::CmpIPredicate::ne:
        return *lhs != *rhs;
      case arith::CmpIPredicate::slt:
      case arith::CmpIPredicate::ult:
        return *lhs < *rhs;
      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::ule:
        return *lhs <= *rhs;
      case arith::CmpIPredicate::sgt:
      case arith::CmpIPredicate::ugt:
        return *lhs > *rhs;
      case arith::CmpIPredicate::sge:
      case arith::CmpIPredicate::uge:
        return *lhs >= *rhs;
      }
    }
    if (auto andOp = value.getDefiningOp<arith::AndIOp>()) {
      auto lhs = evalBool(andOp.getLhs(), coord);
      auto rhs = evalBool(andOp.getRhs(), coord);
      if (lhs && rhs) {
        return *lhs && *rhs;
      }
    }
    if (auto orOp = value.getDefiningOp<arith::OrIOp>()) {
      auto lhs = evalBool(orOp.getLhs(), coord);
      auto rhs = evalBool(orOp.getRhs(), coord);
      if (lhs && rhs) {
        return *lhs || *rhs;
      }
    }
    if (auto xorOp = value.getDefiningOp<arith::XOrIOp>()) {
      auto lhs = evalBool(xorOp.getLhs(), coord);
      auto rhs = evalBool(xorOp.getRhs(), coord);
      if (lhs && rhs) {
        return *lhs != *rhs;
      }
    }
    return std::nullopt;
  }

  Domain getConditionDomain(Value condition) {
    Domain result;
    for (Coord coord : baseDomain.nodes) {
      std::optional<bool> value = evalBool(condition, coord);
      if (!value) {
        return Domain::unknown();
      }
      if (*value) {
        result.nodes.insert(coord);
      }
    }
    return result;
  }

  LogicalResult checkKnownSubset(Operation *op, const Domain &current,
                                 const Domain &allowed, Twine message) {
    if (!current.known) {
      return op->emitOpError()
             << "cannot prove PipeNet guard condition; use coordinate "
                "comparisons over ttl.node(dims=2) or move pipe work under "
                "the matching if_src/if_dst callback";
    }
    if (current.isSubsetOf(allowed)) {
      return success();
    }
    Domain extra = domainSubtract(current, allowed);
    InFlightDiagnostic diag =
        op->emitOpError()
        << message
        << "; guard this DFB block with matching coordinate checks or move "
           "pipe work under if_src/if_dst";
    if (extra.known && !extra.nodes.empty()) {
      Coord example = *extra.nodes.begin();
      diag << " (example node: (" << example.x << ", " << example.y << "))";
    }
    return failure();
  }

  FailureOr<Domain> getPipeNetScopeDomain(PipeNetScopeOp scopeOp) {
    SmallVector<int64_t> ids;
    SmallVector<int64_t> roles;
    if (!readI64Array(scopeOp.getOperation(), kPipeNetIdsAttrName, ids) ||
        !readI64Array(scopeOp.getOperation(), kPipeNetRolesAttrName, roles)) {
      scopeOp.emitOpError() << "requires " << kPipeNetIdsAttrName << " and "
                            << kPipeNetRolesAttrName << " attributes";
      return failure();
    }
    if (ids.size() != roles.size()) {
      scopeOp.emitOpError()
          << "requires equal-length PipeNet id and role arrays";
      return failure();
    }
    Domain result;
    for (auto [pipeNetId, roleValue] : llvm::zip_equal(ids, roles)) {
      if (roleValue != static_cast<int64_t>(PipeRole::Source) &&
          roleValue != static_cast<int64_t>(PipeRole::Destination)) {
        scopeOp.emitOpError() << "has invalid PipeNet role " << roleValue;
        return failure();
      }
      result = domainUnion(
          result,
          getNetRoleDomain(pipeNetId, static_cast<PipeRole>(roleValue)));
    }
    return result;
  }

  LogicalResult verifyCopy(CopyOp copyOp, const Domain &current) {
    if (auto dstPipeType = dyn_cast<PipeType>(copyOp.getDst().getType())) {
      return checkKnownSubset(
          copyOp, current, pipeSourceDomain(dstPipeType),
          "may copy to a pipe outside that pipe's source role");
    }
    if (auto srcPipeType = dyn_cast<PipeType>(copyOp.getSrc().getType())) {
      return checkKnownSubset(
          copyOp, current, pipeDestinationDomain(srcPipeType),
          "may copy from a pipe outside that pipe's destination role");
    }
    return success();
  }

  LogicalResult walkRegion(Region &region, const Domain &current) {
    for (Block &block : region) {
      if (failed(walkBlock(block, current))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult walkBlock(Block &block, const Domain &current) {
    for (Operation &op : block) {
      if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
        Domain condDomain = getConditionDomain(ifOp.getCondition());
        Domain thenDomain = domainIntersect(current, condDomain);
        if (failed(walkRegion(ifOp.getThenRegion(), thenDomain))) {
          return failure();
        }
        if (!ifOp.getElseRegion().empty()) {
          Domain elseDomain =
              domainIntersect(current, domainSubtract(baseDomain, condDomain));
          if (failed(walkRegion(ifOp.getElseRegion(), elseDomain))) {
            return failure();
          }
        }
        continue;
      }

      if (auto ifSrcOp = dyn_cast<IfSrcOp>(&op)) {
        auto pipeType = cast<PipeType>(ifSrcOp.getPipe().getType());
        Domain bodyDomain =
            domainIntersect(current, pipeSourceDomain(pipeType));
        if (failed(walkRegion(ifSrcOp.getBody(), bodyDomain))) {
          return failure();
        }
        continue;
      }

      if (auto ifDstOp = dyn_cast<IfDstOp>(&op)) {
        auto pipeType = cast<PipeType>(ifDstOp.getPipe().getType());
        Domain bodyDomain =
            domainIntersect(current, pipeDestinationDomain(pipeType));
        if (failed(walkRegion(ifDstOp.getBody(), bodyDomain))) {
          return failure();
        }
        continue;
      }

      if (auto scopeOp = dyn_cast<PipeNetScopeOp>(&op)) {
        FailureOr<Domain> scopeDomain = getPipeNetScopeDomain(scopeOp);
        if (failed(scopeDomain)) {
          return failure();
        }
        if (failed(checkKnownSubset(scopeOp, current, *scopeDomain,
                                    "PipeNet scope may execute outside its "
                                    "declared role domain"))) {
          return failure();
        }
        if (failed(walkRegion(scopeOp.getBody(), current))) {
          return failure();
        }
        continue;
      }

      if (auto copyOp = dyn_cast<CopyOp>(&op)) {
        if (failed(verifyCopy(copyOp, current))) {
          return failure();
        }
      } else if (auto pushOp = dyn_cast<CBPushOp>(&op)) {
        if (auto cbIndex = getCBIndex(pushOp.getCb())) {
          cbProducerDomains[*cbIndex] =
              domainUnion(cbProducerDomains[*cbIndex], current);
        }
      } else if (auto waitOp = dyn_cast<CBWaitOp>(&op)) {
        if (auto cbIndex = getCBIndex(waitOp.getCb())) {
          waitUses.push_back({waitOp, current, *cbIndex});
        }
      }

      for (Region &nestedRegion : op.getRegions()) {
        if (failed(walkRegion(nestedRegion, current))) {
          return failure();
        }
      }
    }
    return success();
  }

  LogicalResult verifyCBWaits() {
    for (WaitUse &waitUse : waitUses) {
      auto producerIt = cbProducerDomains.find(waitUse.cbIndex);
      if (producerIt == cbProducerDomains.end()) {
        return waitUse.op.emitOpError()
               << "has no producer domain for DFB index " << waitUse.cbIndex;
      }
      if (failed(checkKnownSubset(
              waitUse.op, waitUse.domain, producerIt->second,
              "may wait on a DFB from nodes where no producer pushes it"))) {
        return failure();
      }
    }
    return success();
  }

  void inlinePipeNetScopes() {
    SmallVector<PipeNetScopeOp> scopes;
    module.walk([&](PipeNetScopeOp scopeOp) { scopes.push_back(scopeOp); });
    for (PipeNetScopeOp scopeOp : scopes) {
      Block &body = scopeOp.getBody().front();
      Operation *scopeOperation = scopeOp.getOperation();
      scopeOperation->getBlock()->getOperations().splice(
          scopeOperation->getIterator(), body.getOperations(), body.begin(),
          body.end());
      scopeOperation->erase();
    }
  }
};

struct TTLVerifyPipeNetGuardsPass
    : impl::TTLVerifyPipeNetGuardsBase<TTLVerifyPipeNetGuardsPass> {
  void runOnOperation() override {
    if (failed(GuardVerifier(getOperation()).run())) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
