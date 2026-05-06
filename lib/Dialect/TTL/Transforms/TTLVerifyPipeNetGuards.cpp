// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Verify PipeNet Guards Pass
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IntegerSet.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
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

enum class PipeRole : int64_t { Source = 0, Destination = 1, Active = 2 };

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

    auto domain = tryBuildBaseDomain();
    if (!domain) {
      return failure();
    }
    baseDomain = *domain;

    validatePipeNetReferences();

    SmallVector<func::FuncOp> kernels;
    module.walk([&](func::FuncOp func) {
      if (func->hasAttr(kKernelThreadAttrName)) {
        kernels.push_back(func);
      }
    });

    for (func::FuncOp func : kernels) {
      if (!llvm::hasSingleElement(func.getBody())) {
        func.emitOpError()
            << "kernel function body has unstructured control flow "
               "(early return, goto-style branching, or multiple basic "
               "blocks); rewrite using structured `if` / `for` / `while` "
               "so PipeNet guard verification can analyze it";
        sawError = true;
        continue;
      }
      walkRegion(func.getBody(), baseDomain);
    }

    // Producer / consumer domains accumulate across every kernel walk
    // before this check — CB indices are module-global, so pairing
    // crosses thread boundaries.
    verifyCBWaits();

    inlinePipeNetScopes();
    return sawError ? failure() : success();
  }

private:
  ModuleOp module;
  bool hasPipes = false;
  bool sawError = false;
  Domain baseDomain;
  std::map<int64_t, Domain> netSourceDomains;
  std::map<int64_t, Domain> netDestinationDomains;
  std::map<int64_t, SmallVector<Location>> pipeNetLocs;
  // User variable name for each PipeNet id, populated from
  // `pipeNetName` on the first `ttl.create_pipe` for that id.
  // `netName()` synthesizes `net_<id>` for ids that didn't carry a
  // name (e.g. handwritten lit IR), so callers can treat the name
  // as always-present.
  std::map<int64_t, std::string> pipeNetNames;
  std::map<int64_t, Domain> cbProducerDomains;
  Operation *lastUnanalyzableOp = nullptr;

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
      pipeNetLocs[pipeNetId].push_back(pipe.getLoc());
      // Take the first non-empty name we see; subsequent pipes in the
      // same net should agree but we don't enforce it.
      auto &name = pipeNetNames[pipeNetId];
      if (name.empty()) {
        if (auto attr = pipe.getPipeNetNameAttr()) {
          name = attr.getValue().str();
        }
      }
    });
  }

  // Identifier used for a PipeNet in every diagnostic. The frontend
  // stamps the user's Python variable name (e.g. `mcast_a_net`);
  // handwritten lit IR may omit it, in which case we synthesize
  // `net_<id>` so callers never have to handle a name-vs-no-name
  // special case. Same string is used in prose ("PipeNet X declared
  // here") and inside code expressions (`X.is_active()`), so the
  // suggested guard is always a valid Python expression.
  std::string netName(int64_t netId) {
    auto it = pipeNetNames.find(netId);
    if (it != pipeNetNames.end() && !it->second.empty()) {
      return it->second;
    }
    return "net_" + std::to_string(netId);
  }

  // Every `is_src` / `is_dst` / `is_active` / `pipenet_scope` op carries a
  // PipeNet id; that id must match some `ttl.create_pipe` in the module.
  //
  // We must check this explicitly because role-domain lookups go through
  // `std::map::operator[]`, which returns an empty `Domain` for missing
  // ids. A scope or guard referencing a bogus id would then trivially
  // satisfy `domain ⊆ ∅` checks (for empty execution domains) and the
  // verifier would silently accept it.
  void validatePipeNetReferences() {
    auto reportUnknownId = [&](Operation *op, int64_t netId) {
      op->emitOpError() << "references unknown PipeNet id " << netId
                        << "; no `ttl.create_pipe` declares this net";
      sawError = true;
    };
    module.walk([&](Operation *op) {
      if (auto isSrc = dyn_cast<IsSrcOp>(op)) {
        if (!pipeNetLocs.count(isSrc.getPipeNetId())) {
          reportUnknownId(op, isSrc.getPipeNetId());
        }
      } else if (auto isDst = dyn_cast<IsDstOp>(op)) {
        if (!pipeNetLocs.count(isDst.getPipeNetId())) {
          reportUnknownId(op, isDst.getPipeNetId());
        }
      } else if (auto isActive = dyn_cast<IsActiveOp>(op)) {
        if (!pipeNetLocs.count(isActive.getPipeNetId())) {
          reportUnknownId(op, isActive.getPipeNetId());
        }
      } else if (isa<PipeNetScopeOp>(op)) {
        SmallVector<int64_t> ids;
        if (readI64Array(op, kPipeNetIdsAttrName, ids)) {
          for (int64_t id : ids) {
            if (!pipeNetLocs.count(id)) {
              reportUnknownId(op, id);
            }
          }
        }
      }
    });
  }

  // Subset checks against an unbounded universe are meaningless, so an
  // explicit launch grid is mandatory.
  std::optional<Domain> tryBuildBaseDomain() {
    auto launchGrid = readLaunchGrid(module);
    if (!launchGrid) {
      module.emitError()
          << "ttl-verify-pipenet-guards requires a `ttl.launch_grid` "
             "module attribute (an i64 array of length 2 with positive "
             "entries)";
      return std::nullopt;
    }
    return fullGridDomain(launchGrid->first, launchGrid->second);
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

  // True if `v`'s expression tree transitively reads a `ttl.core_x` or
  // `ttl.core_y` value. Conditions that don't are uniform across the
  // launch grid and so can't narrow the node domain regardless of their
  // runtime value.
  bool dependsOnCoord(Value v) {
    Operation *op = v.getDefiningOp();
    if (!op) {
      return false;
    }
    if (isa<CoreXOp, CoreYOp>(op)) {
      return true;
    }
    for (Value operand : op->getOperands()) {
      if (dependsOnCoord(operand)) {
        return true;
      }
    }
    return false;
  }

  // Compute upper bounds on the coords that may execute the then- and
  // else-branches of a condition, both intersected with `current`. The
  // abstraction is "branch execution domains" rather than a single
  // "true domain": for a coord-independent predicate (runtime flag,
  // loop iv) neither branch can be narrowed and both inherit `current`.
  //
  // Boolean composition follows the standard upper-bound rules:
  //   A && B: then = A.then ∩ B.then,
  //           else = A.else ∪ (A.then ∩ B.else)
  //   A || B: then = A.then ∪ (A.else ∩ B.then),
  //           else = A.else ∩ B.else
  // These preserve precision for mixed predicates such as
  //   `if runtime_flag and net.is_src(): ...`
  // (then-branch still narrows to the source role; else-branch stays at
  // current).
  struct BranchDomains {
    Domain thenDomain;
    Domain elseDomain;
  };

  BranchDomains exactBranches(const Domain &trueDomain, const Domain &current) {
    return {domainIntersect(current, trueDomain),
            domainIntersect(current, domainSubtract(baseDomain, trueDomain))};
  }

  BranchDomains getBranchDomains(Value condition, const Domain &current) {
    if (auto op = condition.getDefiningOp<IsSrcOp>()) {
      return exactBranches(netSourceDomains[op.getPipeNetId()], current);
    }
    if (auto op = condition.getDefiningOp<IsDstOp>()) {
      return exactBranches(netDestinationDomains[op.getPipeNetId()], current);
    }
    if (auto op = condition.getDefiningOp<IsActiveOp>()) {
      int64_t netId = op.getPipeNetId();
      Domain active =
          domainUnion(netSourceDomains[netId], netDestinationDomains[netId]);
      return exactBranches(active, current);
    }
    if (auto andOp = condition.getDefiningOp<arith::AndIOp>()) {
      auto a = getBranchDomains(andOp.getLhs(), current);
      auto b = getBranchDomains(andOp.getRhs(), current);
      Domain thenDomain = domainIntersect(a.thenDomain, b.thenDomain);
      Domain elseDomain = domainUnion(
          a.elseDomain, domainIntersect(a.thenDomain, b.elseDomain));
      return {thenDomain, elseDomain};
    }
    if (auto orOp = condition.getDefiningOp<arith::OrIOp>()) {
      auto a = getBranchDomains(orOp.getLhs(), current);
      auto b = getBranchDomains(orOp.getRhs(), current);
      Domain thenDomain = domainUnion(
          a.thenDomain, domainIntersect(a.elseDomain, b.thenDomain));
      Domain elseDomain = domainIntersect(a.elseDomain, b.elseDomain);
      return {thenDomain, elseDomain};
    }
    if (!dependsOnCoord(condition)) {
      // Same value at every coord at runtime, but value unknown: either
      // branch could execute on any coord in `current`.
      return {current, current};
    }
    Domain trueDomain;
    for (Coord coord : baseDomain.nodes) {
      std::optional<bool> value = evalBool(condition, coord);
      if (!value) {
        lastUnanalyzableOp = condition.getDefiningOp();
        return {Domain::unknown(), Domain::unknown()};
      }
      if (*value) {
        trueDomain.nodes.insert(coord);
      }
    }
    return exactBranches(trueDomain, current);
  }

  // Each constraint is `expr(dim, sym) >= 0` (or `== 0` for equalities);
  // evaluate them at every coord in the launch grid. Unsupported
  // AffineExprKinds and undefined evaluations (mod / floordiv / ceildiv
  // by zero) propagate as `⊥`; accepting them with a substituted value
  // would silently widen the true-domain.
  Domain getAffineIfDomain(affine::AffineIfOp ifOp) {
    IntegerSet set = ifOp.getIntegerSet();
    auto operands = ifOp.getOperands();
    Domain result;
    SmallVector<int64_t> values(set.getNumInputs(), 0);
    for (Coord coord : baseDomain.nodes) {
      bool resolved = true;
      for (unsigned i = 0; i < set.getNumInputs(); ++i) {
        auto v = evalIndex(operands[i], coord);
        if (!v) {
          resolved = false;
          break;
        }
        values[i] = *v;
      }
      if (!resolved) {
        lastUnanalyzableOp = ifOp;
        return Domain::unknown();
      }
      bool ok = true;
      for (unsigned i = 0; i < set.getNumConstraints(); ++i) {
        AffineExpr expr = set.getConstraint(i);
        bool unsupported = false;
        std::function<int64_t(AffineExpr)> eval = [&](AffineExpr e) -> int64_t {
          if (unsupported) {
            return 0;
          }
          if (auto c = dyn_cast<AffineConstantExpr>(e)) {
            return c.getValue();
          }
          if (auto d = dyn_cast<AffineDimExpr>(e)) {
            return values[d.getPosition()];
          }
          if (auto s = dyn_cast<AffineSymbolExpr>(e)) {
            return values[set.getNumDims() + s.getPosition()];
          }
          if (auto bin = dyn_cast<AffineBinaryOpExpr>(e)) {
            int64_t lhs = eval(bin.getLHS());
            int64_t rhs = eval(bin.getRHS());
            switch (bin.getKind()) {
            case AffineExprKind::Add:
              return lhs + rhs;
            case AffineExprKind::Mul:
              return lhs * rhs;
            case AffineExprKind::Mod:
              if (rhs == 0) {
                unsupported = true;
                return 0;
              }
              return lhs % rhs;
            case AffineExprKind::FloorDiv:
              if (rhs == 0) {
                unsupported = true;
                return 0;
              }
              return lhs / rhs - (lhs % rhs && (lhs ^ rhs) < 0 ? 1 : 0);
            case AffineExprKind::CeilDiv:
              if (rhs == 0) {
                unsupported = true;
                return 0;
              }
              return lhs / rhs + (lhs % rhs && (lhs ^ rhs) > 0 ? 1 : 0);
            default:
              unsupported = true;
              return 0;
            }
          }
          unsupported = true;
          return 0;
        };
        int64_t v = eval(expr);
        if (unsupported) {
          lastUnanalyzableOp = ifOp;
          return Domain::unknown();
        }
        if (set.isEq(i) ? v != 0 : v < 0) {
          ok = false;
          break;
        }
      }
      if (ok) {
        result.nodes.insert(coord);
      }
    }
    return result;
  }

  // Render the runtime predicate that matches the verifier's role
  // domain for `roles`. Same input semantics as the verifier check
  // (which unions all declared roles), so the rendered expression
  // joins clauses with ` or `:
  //
  //   net_0.is_src()                    (one net, one role)
  //   net_0.is_active()                 (one net, both src and dst → collapse)
  //   net_0.is_dst() or net_1.is_src()  (different nets, joined by union)
  //
  // No leading `if `, no trailing `:`, no surrounding backticks —
  // callers wrap as needed (the standalone `suggested guard:` note
  // adds backticks; the in-code-span use in primary messages does
  // not, since it already sits inside a backtick code span).
  std::string
  formatGuardExpression(ArrayRef<std::pair<int64_t, PipeRole>> roles) {
    // Group roles by net id, preserving first-seen order. Within a net,
    // collect a flag set of {src, dst, active}.
    SmallVector<int64_t> orderedIds;
    std::map<int64_t, std::array<bool, 3>> rolesByNet;
    for (auto [id, role] : roles) {
      auto [it, inserted] =
          rolesByNet.try_emplace(id, std::array<bool, 3>{false, false, false});
      if (inserted) {
        orderedIds.push_back(id);
      }
      it->second[static_cast<size_t>(role)] = true;
    }

    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    bool first = true;
    for (int64_t id : orderedIds) {
      auto &flags = rolesByNet[id];
      bool hasSrc = flags[static_cast<size_t>(PipeRole::Source)];
      bool hasDst = flags[static_cast<size_t>(PipeRole::Destination)];
      bool hasActive = flags[static_cast<size_t>(PipeRole::Active)];

      // src ∨ dst is exactly is_active; collapse so we don't emit the
      // ambiguous `net_0.is_src() or net_0.is_dst()` form.
      bool collapsedToActive = hasActive || (hasSrc && hasDst);
      auto emit = [&](StringRef method) {
        if (!first) {
          os << " or ";
        }
        first = false;
        os << netName(id) << "." << method << "()";
      };
      if (collapsedToActive) {
        emit("is_active");
      } else if (hasSrc) {
        emit("is_src");
      } else if (hasDst) {
        emit("is_dst");
      }
    }
    return std::move(os.str());
  }

  void attachWitnessNote(InFlightDiagnostic &diag, const Domain &extra) {
    if (extra.known && !extra.nodes.empty()) {
      Coord example = *extra.nodes.begin();
      diag.attachNote() << "example node where the guard does not hold: "
                        << "core_x=" << example.x << ", core_y=" << example.y;
    }
  }

  void attachPipeNetNotes(InFlightDiagnostic &diag, ArrayRef<int64_t> netIds) {
    for (int64_t netId : netIds) {
      auto it = pipeNetLocs.find(netId);
      if (it == pipeNetLocs.end() || it->second.empty()) {
        continue;
      }
      diag.attachNote(it->second.front())
          << "PipeNet " << netName(netId) << " declared here";
    }
  }

  void checkKnownSubset(Operation *op, const Domain &current,
                        const Domain &allowed, Twine primaryMessage,
                        ArrayRef<std::pair<int64_t, PipeRole>> roles = {}) {
    if (!current.known) {
      auto diag = op->emitOpError()
                  << "could not statically analyze the PipeNet guard "
                     "around this op; rewrite using `net.is_src()` / "
                     "`net.is_dst()` / `net.is_active()`, or compare "
                     "`ttl.node(dims=2)` coordinates against integer "
                     "constants";
      if (lastUnanalyzableOp) {
        diag.attachNote(lastUnanalyzableOp->getLoc())
            << "this expression is not statically analyzable";
      }
      sawError = true;
      return;
    }
    if (current.isSubsetOf(allowed)) {
      return;
    }
    Domain extra = domainSubtract(current, allowed);
    auto diag = op->emitOpError() << primaryMessage;
    attachWitnessNote(diag, extra);
    SmallVector<int64_t> ids;
    for (auto &p : roles) {
      ids.push_back(p.first);
    }
    attachPipeNetNotes(diag, ids);
    // The suggested guard is already embedded in `primaryMessage`
    // (callers that pass roles render the same `is_src/is_dst/is_active`
    // expression inline). Repeating it as a note would surface the
    // same string twice in the user-facing diagnostic.
    sawError = true;
  }

  // Returns std::nullopt and emits a diagnostic if the scope's attributes
  // are missing or malformed.
  std::optional<std::pair<Domain, SmallVector<std::pair<int64_t, PipeRole>>>>
  getPipeNetScopeDomain(PipeNetScopeOp scopeOp) {
    SmallVector<int64_t> ids;
    SmallVector<int64_t> roles;
    if (!readI64Array(scopeOp.getOperation(), kPipeNetIdsAttrName, ids) ||
        !readI64Array(scopeOp.getOperation(), kPipeNetRolesAttrName, roles)) {
      scopeOp.emitOpError() << "requires `" << kPipeNetIdsAttrName << "` and `"
                            << kPipeNetRolesAttrName << "` attributes";
      sawError = true;
      return std::nullopt;
    }
    if (ids.size() != roles.size()) {
      scopeOp.emitOpError()
          << "requires equal-length PipeNet id and role arrays";
      sawError = true;
      return std::nullopt;
    }
    Domain result;
    SmallVector<std::pair<int64_t, PipeRole>> declaredRoles;
    for (auto [pipeNetId, roleValue] : llvm::zip_equal(ids, roles)) {
      if (roleValue != static_cast<int64_t>(PipeRole::Source) &&
          roleValue != static_cast<int64_t>(PipeRole::Destination)) {
        scopeOp.emitOpError() << "has invalid PipeNet role " << roleValue
                              << " (expected 0=src or 1=dst)";
        sawError = true;
        return std::nullopt;
      }
      auto role = static_cast<PipeRole>(roleValue);
      result = domainUnion(result, getNetRoleDomain(pipeNetId, role));
      declaredRoles.emplace_back(pipeNetId, role);
    }
    return std::make_pair(result, declaredRoles);
  }

  void verifyCopy(CopyOp copyOp, const Domain &current) {
    if (auto dstPipeType = dyn_cast<PipeType>(copyOp.getDst().getType())) {
      int64_t netId = dstPipeType.getPipeNetId();
      std::string name = netName(netId);
      std::string msg;
      llvm::raw_string_ostream(msg)
          << "this `ttl.copy(buffer, pipe)` sends data on PipeNet " << name
          << " from a node that is not a source of any pipe in that net; "
             "wrap the copy in `"
          << name << ".if_src(...)` or guard with `if " << name
          << ".is_src(): ...`";
      checkKnownSubset(copyOp, current, pipeSourceDomain(dstPipeType), msg,
                       {{netId, PipeRole::Source}});
      return;
    }
    if (auto srcPipeType = dyn_cast<PipeType>(copyOp.getSrc().getType())) {
      int64_t netId = srcPipeType.getPipeNetId();
      std::string name = netName(netId);
      std::string msg;
      llvm::raw_string_ostream(msg)
          << "this `ttl.copy(pipe, buffer)` receives data from PipeNet " << name
          << " on a node that is not a destination of any pipe in that "
             "net; wrap the copy in `"
          << name << ".if_dst(...)` or guard with `if " << name
          << ".is_dst(): ...`";
      checkKnownSubset(copyOp, current, pipeDestinationDomain(srcPipeType), msg,
                       {{netId, PipeRole::Destination}});
    }
  }

  void walkRegion(Region &region, const Domain &current) {
    for (Block &block : region) {
      walkBlock(block, current);
    }
  }

  void walkBlock(Block &block, const Domain &current) {
    for (Operation &op : block) {
      if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
        auto [thenDomain, elseDomain] =
            getBranchDomains(ifOp.getCondition(), current);
        walkRegion(ifOp.getThenRegion(), thenDomain);
        if (!ifOp.getElseRegion().empty()) {
          walkRegion(ifOp.getElseRegion(), elseDomain);
        }
        continue;
      }

      if (auto affineIf = dyn_cast<affine::AffineIfOp>(&op)) {
        Domain condDomain = getAffineIfDomain(affineIf);
        Domain thenDomain = domainIntersect(current, condDomain);
        walkRegion(affineIf.getThenRegion(), thenDomain);
        if (!affineIf.getElseRegion().empty()) {
          Domain elseDomain =
              domainIntersect(current, domainSubtract(baseDomain, condDomain));
          walkRegion(affineIf.getElseRegion(), elseDomain);
        }
        continue;
      }

      if (auto ifSrcOp = dyn_cast<IfSrcOp>(&op)) {
        auto pipeType = cast<PipeType>(ifSrcOp.getPipe().getType());
        Domain bodyDomain =
            domainIntersect(current, pipeSourceDomain(pipeType));
        walkRegion(ifSrcOp.getBody(), bodyDomain);
        continue;
      }

      if (auto ifDstOp = dyn_cast<IfDstOp>(&op)) {
        auto pipeType = cast<PipeType>(ifDstOp.getPipe().getType());
        Domain bodyDomain =
            domainIntersect(current, pipeDestinationDomain(pipeType));
        walkRegion(ifDstOp.getBody(), bodyDomain);
        continue;
      }

      if (auto scopeOp = dyn_cast<PipeNetScopeOp>(&op)) {
        auto scope = getPipeNetScopeDomain(scopeOp);
        if (!scope) {
          continue;
        }
        // Collect the unique PipeNet ids referenced by this scope so
        // the diagnostic names every net the user must guard against.
        SmallVector<int64_t> ids;
        for (auto &p : scope->second) {
          if (!llvm::is_contained(ids, p.first)) {
            ids.push_back(p.first);
          }
        }
        std::string msg;
        {
          llvm::raw_string_ostream os(msg);
          os << "this region exchanges data on PipeNet";
          if (ids.size() != 1) {
            os << "s";
          }
          os << " ";
          llvm::interleaveComma(ids, os,
                                [&](int64_t id) { os << netName(id); });
          os << " on launched nodes that are not part of "
             << (ids.size() == 1 ? "that net" : "those nets")
             << "; wrap the surrounding work in `if "
             << formatGuardExpression(scope->second)
             << ": ...` so non-participating nodes skip it";
        }
        checkKnownSubset(scopeOp, current, scope->first, msg, scope->second);
        walkRegion(scopeOp.getBody(), current);
        continue;
      }

      if (auto copyOp = dyn_cast<CopyOp>(&op)) {
        verifyCopy(copyOp, current);
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
        walkRegion(nestedRegion, current);
      }
    }
  }

  void verifyCBWaits() {
    for (WaitUse &use : waitUses) {
      auto it = cbProducerDomains.find(use.cbIndex);
      if (it == cbProducerDomains.end()) {
        use.op.emitOpError()
            << "this `cb_wait` reads from a dataflow buffer that no other "
               "thread fills; check that another `@ttl.compute()` or "
               "`@ttl.datamovement()` thread reserves and pushes the same "
               "buffer";
        sawError = true;
        continue;
      }
      checkKnownSubset(use.op, use.domain, it->second,
                       "this `cb_wait` runs on launched nodes where no "
                       "thread pushes data to the buffer (would deadlock); "
                       "guard the wait with the same `if net.is_active(): "
                       "...` predicate the producer uses");
    }
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
