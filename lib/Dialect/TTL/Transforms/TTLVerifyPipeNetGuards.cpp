// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Verify that PipeNet-coupled operations execute only on launch nodes whose
// PipeNet roles permit them. The analysis is a `DenseForwardDataFlowAnalysis`
// whose lattice is the set of launch coordinates that can reach each program
// point. Predicate-bearing region ops (`scf.if`, `affine.if`, `ttl.if_src`,
// `ttl.if_dst`, `ttl.pipenet_scope`) narrow that set on region entry; pipe-
// coupled ops are checked against the narrowed set.
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IntegerSet.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <optional>
#include <set>
#include <tuple>

#define DEBUG_TYPE "ttl-verify-pipenet-guards"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLVERIFYPIPENETGUARDS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kLaunchGridAttrName = "ttl.launch_grid";
constexpr llvm::StringLiteral kPipeNetIdsAttrName = "ttl.pipe_net_ids";
constexpr llvm::StringLiteral kPipeNetRolesAttrName = "ttl.pipe_net_roles";
constexpr unsigned kMaxPipeScheduleCycleNotes = 8;

// A 2D coordinate representing a launch node.
struct Coord {
  int64_t x = 0;
  int64_t y = 0;

  bool operator<(const Coord &rhs) const {
    return std::tie(x, y) < std::tie(rhs.x, rhs.y);
  }
  bool operator==(const Coord &rhs) const { return x == rhs.x && y == rhs.y; }
};

// A set of launch nodes (coordinates). `known` is false when the verifier could
// not determine the domain.
struct Domain {
  bool known = true;
  std::set<Coord> nodes;

  static Domain unknown() { return {/*known=*/false, {}}; }

  bool isSubsetOf(const Domain &rhs) const {
    if (!known || !rhs.known) {
      return false;
    }
    return std::includes(rhs.nodes.begin(), rhs.nodes.end(), nodes.begin(),
                         nodes.end());
  }

  bool operator==(const Domain &rhs) const {
    return known == rhs.known && nodes == rhs.nodes;
  }
};

// Set union of two domains; result is unknown if either input is unknown.
Domain domainUnion(const Domain &lhs, const Domain &rhs) {
  if (!lhs.known || !rhs.known) {
    return Domain::unknown();
  }
  Domain result;
  std::set_union(lhs.nodes.begin(), lhs.nodes.end(), rhs.nodes.begin(),
                 rhs.nodes.end(),
                 std::inserter(result.nodes, result.nodes.end()));
  return result;
}

// Set intersection of two domains; result is unknown if either input is
// unknown.
Domain domainIntersect(const Domain &lhs, const Domain &rhs) {
  if (!lhs.known || !rhs.known) {
    return Domain::unknown();
  }
  Domain result;
  std::set_intersection(lhs.nodes.begin(), lhs.nodes.end(), rhs.nodes.begin(),
                        rhs.nodes.end(),
                        std::inserter(result.nodes, result.nodes.end()));
  return result;
}

// Set difference `lhs \ rhs`; result is unknown if either input is unknown.
Domain domainSubtract(const Domain &lhs, const Domain &rhs) {
  if (!lhs.known || !rhs.known) {
    return Domain::unknown();
  }
  Domain result;
  std::set_difference(lhs.nodes.begin(), lhs.nodes.end(), rhs.nodes.begin(),
                      rhs.nodes.end(),
                      std::inserter(result.nodes, result.nodes.end()));
  return result;
}

// Domain containing every coord in the [0, gridX) x [0, gridY) launch grid.
Domain fullGridDomain(int64_t gridX, int64_t gridY) {
  Domain result;
  for (int64_t x = 0; x < gridX; ++x) {
    for (int64_t y = 0; y < gridY; ++y) {
      result.nodes.insert({x, y});
    }
  }
  return result;
}

// Singleton domain containing the pipe's source coord.
Domain pipeSourceDomain(PipeType pipeType) {
  Domain result;
  result.nodes.insert({pipeType.getSrcX(), pipeType.getSrcY()});
  return result;
}

// Rectangular domain spanned by the pipe's destination range.
Domain pipeDestinationDomain(PipeType pipeType) {
  Domain result;
  for (int64_t x = pipeType.getDstStartX(); x <= pipeType.getDstEndX(); ++x) {
    for (int64_t y = pipeType.getDstStartY(); y <= pipeType.getDstEndY(); ++y) {
      result.nodes.insert({x, y});
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Generic helpers.
//===----------------------------------------------------------------------===//

// Read an i64 array attribute from `op`, accepting either a DenseI64ArrayAttr
// or an ArrayAttr of IntegerAttrs. Returns false if the attribute is missing
// or has an incompatible element type.
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
    auto intAttr = mlir::dyn_cast<IntegerAttr>(attr);
    if (!intAttr) {
      return false;
    }
    values.push_back(intAttr.getInt());
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Module state collected before the analysis runs and updated during it.
//===----------------------------------------------------------------------===//

struct WaitUse {
  CBWaitOp op;
  Domain domain;
  int64_t cbIndex;
};

bool isPipeReceiveCopy(CopyOp copyOp) {
  return mlir::isa<PipeType>(copyOp.getSrc().getType()) &&
         getAttachedCB(copyOp.getDst());
}

std::optional<CopyOp> findDefiningPipeReceiveCopy(Value value) {
  llvm::SmallPtrSet<Value, 16> seen;
  return traceTransferHandleSource<std::optional<CopyOp>>(
      value,
      [](Value source) {
        auto copyOp = source.getDefiningOp<CopyOp>();
        if (copyOp && isPipeReceiveCopy(copyOp)) {
          return std::optional<CopyOp>(copyOp);
        }
        return std::optional<CopyOp>();
      },
      seen);
}

enum class PipeEventKind { Send, ReceivePost, ReceiveWait };

struct PipeEvent {
  Operation *op = nullptr;
  PipeType pipeType;
  PipeEventKind kind;
  Domain domain;
};

struct ModuleState;
void checkKnownSubset(Operation *op, const Domain &current,
                      const Domain &allowed, Operation *unanalyzableOp,
                      Twine primaryMessage,
                      ArrayRef<std::pair<int64_t, PipeRole>> roles,
                      ModuleState &state);

struct ModuleState {
  Domain baseDomain;
  llvm::DenseMap<int64_t, Domain> netSourceDomains;
  llvm::DenseMap<int64_t, Domain> netDestinationDomains;
  llvm::DenseMap<int64_t, SmallVector<Location>> pipeNetLocs;
  llvm::DenseMap<int64_t, std::string> pipeNetNames;
  llvm::DenseMap<int64_t, Domain> cbProducerDomains;
  SmallVector<WaitUse> waitUses;
  SmallVector<PipeEvent> pipeEvents;
  llvm::DenseMap<Operation *, unsigned> pipeEventIndices;
  bool sawError = false;

  bool hasPipes() const { return !pipeNetLocs.empty(); }

  // Synthesize `net_<id>` for IR without the frontend's `pipeNetName` attr
  // (typically handwritten lit cases) so every diagnostic names PipeNets the
  // same way.
  std::string netName(int64_t netId) const {
    auto it = pipeNetNames.find(netId);
    if (it != pipeNetNames.end() && !it->second.empty()) {
      return it->second;
    }
    return "net_" + std::to_string(netId);
  }

  Domain getRoleDomain(int64_t netId, PipeRole role) const {
    if (role == PipeRole::Source) {
      auto it = netSourceDomains.find(netId);
      return it == netSourceDomains.end() ? Domain{} : it->second;
    }
    if (role == PipeRole::Destination) {
      auto it = netDestinationDomains.find(netId);
      return it == netDestinationDomains.end() ? Domain{} : it->second;
    }
    Domain src;
    Domain dst;
    if (auto it = netSourceDomains.find(netId); it != netSourceDomains.end()) {
      src = it->second;
    }
    if (auto it = netDestinationDomains.find(netId);
        it != netDestinationDomains.end()) {
      dst = it->second;
    }
    return domainUnion(src, dst);
  }

  LogicalResult initialize(ModuleOp module) {
    module.walk([&](CreatePipeOp pipe) {
      PipeType pipeType = mlir::cast<PipeType>(pipe.getResult().getType());
      int64_t pipeNetId = pipeType.getPipeNetId();
      netSourceDomains[pipeNetId] =
          domainUnion(netSourceDomains[pipeNetId], pipeSourceDomain(pipeType));
      netDestinationDomains[pipeNetId] = domainUnion(
          netDestinationDomains[pipeNetId], pipeDestinationDomain(pipeType));
      pipeNetLocs[pipeNetId].push_back(pipe.getLoc());
      auto &name = pipeNetNames[pipeNetId];
      if (name.empty()) {
        if (auto attr = pipe.getPipeNetNameAttr()) {
          name = attr.getValue().str();
        }
      }
    });

    if (!hasPipes()) {
      return success();
    }

    SmallVector<int64_t> launchGrid;
    if (!readI64Array(module.getOperation(), kLaunchGridAttrName, launchGrid) ||
        launchGrid.size() != 2 || launchGrid[0] <= 0 || launchGrid[1] <= 0) {
      module.emitError()
          << "ttl-verify-pipenet-guards requires a `ttl.launch_grid` "
             "module attribute (an i64 array of length 2 with positive "
             "entries)";
      return failure();
    }
    baseDomain = fullGridDomain(launchGrid[0], launchGrid[1]);
    return success();
  }

  void recordPipeEvent(CopyOp copyOp, const Domain &domain) {
    PipeEvent event;
    event.op = copyOp.getOperation();
    if (auto pipeType = mlir::dyn_cast<PipeType>(copyOp.getDst().getType())) {
      event.pipeType = pipeType;
      event.kind = PipeEventKind::Send;
      event.domain = domainIntersect(domain, pipeSourceDomain(pipeType));
    } else if (auto pipeType =
                   mlir::dyn_cast<PipeType>(copyOp.getSrc().getType())) {
      if (!isPipeReceiveCopy(copyOp)) {
        return;
      }
      event.pipeType = pipeType;
      event.kind = PipeEventKind::ReceivePost;
      event.domain = domainIntersect(domain, pipeDestinationDomain(pipeType));
    } else {
      return;
    }

    auto [it, inserted] =
        pipeEventIndices.try_emplace(copyOp.getOperation(), pipeEvents.size());
    if (inserted) {
      pipeEvents.push_back(event);
      return;
    }
    pipeEvents[it->second] = event;
  }

  void recordPipeWaitEvent(WaitOp waitOp, const Domain &domain,
                           Operation *unanalyzableOp) {
    std::optional<CopyOp> copyOp = findDefiningPipeReceiveCopy(waitOp.getXf());
    if (!copyOp.has_value()) {
      return;
    }
    auto pipeType = mlir::cast<PipeType>(copyOp->getSrc().getType());

    int64_t netId = pipeType.getPipeNetId();
    std::string name = netName(netId);
    std::string msg;
    llvm::raw_string_ostream(msg)
        << "this `ttl.wait` waits for a pipe receive on launched nodes "
           "that are not destinations of PipeNet "
        << name << "; keep the wait under the same `if " << name
        << ".is_dst(): ...` or `" << name
        << ".if_dst(...)` guard as the receive copy";
    checkKnownSubset(waitOp, domain, pipeDestinationDomain(pipeType),
                     unanalyzableOp, msg, {{netId, PipeRole::Destination}},
                     *this);
    if (sawError) {
      return;
    }

    PipeEvent event;
    event.op = waitOp.getOperation();
    event.pipeType = pipeType;
    event.kind = PipeEventKind::ReceiveWait;
    event.domain = domainIntersect(domain, pipeDestinationDomain(pipeType));

    auto [it, inserted] =
        pipeEventIndices.try_emplace(waitOp.getOperation(), pipeEvents.size());
    if (inserted) {
      pipeEvents.push_back(event);
      return;
    }
    pipeEvents[it->second] = event;
  }
};

//===----------------------------------------------------------------------===//
// Predicate decomposition.
//===----------------------------------------------------------------------===//

struct DomainResult {
  Domain domain;
  Operation *unanalyzableOp = nullptr;
};

// Evaluate an index/integer-typed `Value` at the given launch coord by
// substituting `core_x`/`core_y` and folding through arith add/sub/mul and
// index_cast. Returns nullopt if any subexpression cannot be folded.
std::optional<int64_t> evalIndex(Value value, Coord coord) {
  if (value.getDefiningOp<CoreXOp>()) {
    return coord.x;
  }
  if (value.getDefiningOp<CoreYOp>()) {
    return coord.y;
  }
  if (auto constant = getConstantIntValue(value)) {
    return *constant;
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

// Evaluate an i1-typed predicate `Value` at the given launch coord by
// folding through cmpi (over `evalIndex` operands) and the boolean
// connectives andi/ori/xori. Returns nullopt for non-foldable subexpressions.
std::optional<bool> evalBool(Value value, Coord coord) {
  if (value.getType().isInteger(1)) {
    if (auto constant = getConstantIntValue(value)) {
      return *constant != 0;
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

// Memoize so a shared SSA value reached via multiple operand chains is
// visited once. Without the cache the walk is exponential in the size of
// any DAG above the condition.
bool dependsOnCoord(Value v, llvm::DenseMap<Value, bool> &cache) {
  if (auto it = cache.find(v); it != cache.end()) {
    return it->second;
  }
  Operation *op = v.getDefiningOp();
  bool result = false;
  if (op) {
    if (mlir::isa<CoreXOp, CoreYOp>(op)) {
      result = true;
    } else {
      for (Value operand : op->getOperands()) {
        if (dependsOnCoord(operand, cache)) {
          result = true;
          break;
        }
      }
    }
  }
  cache[v] = result;
  return result;
}

// Per-coord evaluate the constraints of an `affine.if` IntegerSet. Pack all
// constraints into one AffineMap (one result per constraint) so each coord
// requires a single `constantFold` rather than one per constraint.
DomainResult getAffineIfDomain(affine::AffineIfOp ifOp,
                               const Domain &baseDomain) {
  IntegerSet set = ifOp.getIntegerSet();
  ValueRange operands = ifOp.getOperands();
  MLIRContext *ctx = ifOp.getContext();

  SmallVector<AffineExpr> constraintExprs;
  constraintExprs.reserve(set.getNumConstraints());
  for (unsigned i = 0; i < set.getNumConstraints(); ++i) {
    constraintExprs.push_back(set.getConstraint(i));
  }
  AffineMap map = AffineMap::get(set.getNumDims(), set.getNumSymbols(),
                                 constraintExprs, ctx);

  Domain result;
  SmallVector<Attribute> operandConstants(set.getNumInputs());
  for (Coord coord : baseDomain.nodes) {
    bool resolved = true;
    for (unsigned i = 0; i < set.getNumInputs(); ++i) {
      auto v = evalIndex(operands[i], coord);
      if (!v) {
        resolved = false;
        break;
      }
      operandConstants[i] = IntegerAttr::get(IndexType::get(ctx), *v);
    }
    if (!resolved) {
      return {Domain::unknown(), ifOp};
    }
    SmallVector<Attribute> folded;
    if (failed(map.constantFold(operandConstants, folded))) {
      return {Domain::unknown(), ifOp};
    }
    bool ok = true;
    for (unsigned i = 0; i < set.getNumConstraints(); ++i) {
      auto intAttr = mlir::dyn_cast<IntegerAttr>(folded[i]);
      if (!intAttr) {
        return {Domain::unknown(), ifOp};
      }
      int64_t v = intAttr.getInt();
      if (set.isEq(i) ? v != 0 : v < 0) {
        ok = false;
        break;
      }
    }
    if (ok) {
      result.nodes.insert(coord);
    }
  }
  return {result, nullptr};
}

// Walk a Location to find a FileLineColLoc, recursing into FusedLoc (first
// inner location) and CallSiteLoc (callee, then caller). Returns null if
// none is reachable.
static FileLineColLoc findFileLineColLoc(Location loc) {
  if (auto fl = mlir::dyn_cast<FileLineColLoc>(loc)) {
    return fl;
  }
  if (auto fused = mlir::dyn_cast<FusedLoc>(loc)) {
    for (Location inner : fused.getLocations()) {
      if (auto fl = findFileLineColLoc(inner)) {
        return fl;
      }
    }
  }
  if (auto call = mlir::dyn_cast<CallSiteLoc>(loc)) {
    if (auto fl = findFileLineColLoc(call.getCallee())) {
      return fl;
    }
    if (auto fl = findFileLineColLoc(call.getCaller())) {
      return fl;
    }
  }
  return {};
}

// Pick the op whose source Location sorts earlier, so diagnostic notes
// are deterministic across runs even when the dataflow solver visits
// operands in varying order. Either argument may be null.
Operation *pickEarlierBySourceLoc(Operation *lhs, Operation *rhs) {
  if (!lhs) {
    return rhs;
  }
  if (!rhs) {
    return lhs;
  }
  // Comparing Location attributes directly is pointer-based and not
  // stable across runs (Attribute uniquing order depends on allocation),
  // so compare (filename, line, column) extracted from a FileLineColLoc.
  FileLineColLoc lfl = findFileLineColLoc(lhs->getLoc());
  FileLineColLoc rfl = findFileLineColLoc(rhs->getLoc());
  if (lfl && rfl) {
    auto key = [](FileLineColLoc fl) {
      return std::tuple(fl.getFilename().getValue(), fl.getLine(),
                        fl.getColumn());
    };
    return key(lfl) <= key(rfl) ? lhs : rhs;
  }
  // Fallback for locations with no reachable FileLineColLoc (rare).
  std::string lhsStr;
  std::string rhsStr;
  llvm::raw_string_ostream(lhsStr) << lhs->getLoc();
  llvm::raw_string_ostream(rhsStr) << rhs->getLoc();
  return lhsStr <= rhsStr ? lhs : rhs;
}

struct BranchDomains {
  Domain thenDomain;
  Domain elseDomain;
  Operation *unanalyzableOp = nullptr;
};

// Split `current` into the then/else domains for a predicate whose statically
// known true-set is `trueDomain` (relative to `baseDomain`).
BranchDomains exactBranches(const Domain &trueDomain, const Domain &current,
                            const Domain &baseDomain) {
  return {domainIntersect(current, trueDomain),
          domainIntersect(current, domainSubtract(baseDomain, trueDomain))};
}

// Recursively decompose an scf.if `condition` into the then/else launch-coord
// domains. PipeNet predicates use their declared role domain directly; arith
// and/or compose via the standard branch algebra; coord-independent
// subexpressions widen to `current` on both sides; coord-dependent leaves are
// evaluated per-coord via `evalBool`. `coordCache` memoizes the
// coord-dependence walk so that values used by multiple operands are visited
// once, keeping the recursion linear in the number of SSA values reachable
// from `condition`.
BranchDomains getBranchDomainsImpl(Value condition, const Domain &current,
                                   const ModuleState &state,
                                   llvm::DenseMap<Value, bool> &coordCache) {
  if (auto pred = condition.getDefiningOp<PipeNetPredicateOpInterface>()) {
    Domain roleDomain = state.getRoleDomain(pred.getReferencedPipeNetId(),
                                            pred.getReferencedRole());
    return exactBranches(roleDomain, current, state.baseDomain);
  }
  if (auto andOp = condition.getDefiningOp<arith::AndIOp>()) {
    BranchDomains a =
        getBranchDomainsImpl(andOp.getLhs(), current, state, coordCache);
    BranchDomains b =
        getBranchDomainsImpl(andOp.getRhs(), current, state, coordCache);
    Operation *unanalyzable =
        pickEarlierBySourceLoc(a.unanalyzableOp, b.unanalyzableOp);
    return {
        domainIntersect(a.thenDomain, b.thenDomain),
        domainUnion(a.elseDomain, domainIntersect(a.thenDomain, b.elseDomain)),
        unanalyzable};
  }
  if (auto orOp = condition.getDefiningOp<arith::OrIOp>()) {
    BranchDomains a =
        getBranchDomainsImpl(orOp.getLhs(), current, state, coordCache);
    BranchDomains b =
        getBranchDomainsImpl(orOp.getRhs(), current, state, coordCache);
    Operation *unanalyzable =
        pickEarlierBySourceLoc(a.unanalyzableOp, b.unanalyzableOp);
    return {
        domainUnion(a.thenDomain, domainIntersect(a.elseDomain, b.thenDomain)),
        domainIntersect(a.elseDomain, b.elseDomain), unanalyzable};
  }
  if (!dependsOnCoord(condition, coordCache)) {
    // Same value at every coord at runtime, but unknown statically: either
    // branch could execute on any coord in `current`.
    return {current, current};
  }
  Domain trueDomain;
  for (Coord coord : state.baseDomain.nodes) {
    std::optional<bool> value = evalBool(condition, coord);
    if (!value) {
      return {Domain::unknown(), Domain::unknown(), condition.getDefiningOp()};
    }
    if (*value) {
      trueDomain.nodes.insert(coord);
    }
  }
  BranchDomains result = exactBranches(trueDomain, current, state.baseDomain);
  return {result.thenDomain, result.elseDomain, nullptr};
}

// Public entry point for `getBranchDomainsImpl`; owns the per-call
// coord-dependence cache.
BranchDomains getBranchDomains(Value condition, const Domain &current,
                               const ModuleState &state) {
  llvm::DenseMap<Value, bool> coordCache;
  return getBranchDomainsImpl(condition, current, state, coordCache);
}

//===----------------------------------------------------------------------===//
// Diagnostic helpers.
//===----------------------------------------------------------------------===//

// Render the verifier's role domain back as a runtime predicate string.
// Examples:
//   net_0.is_src()                    (one net, one role)
//   net_0.is_active()                 (one net, src and dst both seen)
//   net_0.is_dst() or net_1.is_src()  (different nets)
//
// Input roles are only `Source` or `Destination` (from `pipenet_scope`);
// `is_active` is synthesized when a net has both.
std::string formatGuardExpression(ArrayRef<std::pair<int64_t, PipeRole>> roles,
                                  const ModuleState &state) {
  SmallVector<int64_t> orderedIds;
  llvm::DenseMap<int64_t, std::pair<bool, bool>> rolesByNet; // (hasSrc, hasDst)
  for (auto [id, role] : roles) {
    auto [it, inserted] = rolesByNet.try_emplace(id, std::pair{false, false});
    if (inserted) {
      orderedIds.push_back(id);
    }
    if (role == PipeRole::Source) {
      it->second.first = true;
    } else {
      it->second.second = true;
    }
  }

  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  bool first = true;
  for (int64_t id : orderedIds) {
    auto [hasSrc, hasDst] = rolesByNet[id];
    if (!first) {
      os << " or ";
    }
    first = false;
    StringRef method =
        (hasSrc && hasDst) ? "is_active" : (hasSrc ? "is_src" : "is_dst");
    os << state.netName(id) << "." << method << "()";
  }
  return buffer;
}

// Emit an op error when `current` is not a subset of `allowed`. Attaches an
// example offending coord, the unanalyzable predicate location (if any), and
// declaration notes for each named PipeNet role.
void checkKnownSubset(Operation *op, const Domain &current,
                      const Domain &allowed, Operation *unanalyzableOp,
                      Twine primaryMessage,
                      ArrayRef<std::pair<int64_t, PipeRole>> roles,
                      ModuleState &state) {
  if (!current.known) {
    auto diag = op->emitOpError()
                << "could not statically analyze the PipeNet guard "
                   "around this op; rewrite using `net.is_src()` / "
                   "`net.is_dst()` / `net.is_active()`, or compare "
                   "`ttl.node(dims=2)` coordinates against integer "
                   "constants";
    if (unanalyzableOp) {
      diag.attachNote(unanalyzableOp->getLoc())
          << "this expression is not statically analyzable";
    }
    state.sawError = true;
    return;
  }
  if (current.isSubsetOf(allowed)) {
    return;
  }
  Domain extra = domainSubtract(current, allowed);
  auto diag = op->emitOpError() << primaryMessage;
  if (extra.known && !extra.nodes.empty()) {
    Coord example = *extra.nodes.begin();
    diag.attachNote() << "example node where the guard does not hold: "
                      << "core_x=" << example.x << ", core_y=" << example.y;
  }
  for (auto &p : roles) {
    auto it = state.pipeNetLocs.find(p.first);
    if (it == state.pipeNetLocs.end() || it->second.empty()) {
      continue;
    }
    diag.attachNote(it->second.front())
        << "PipeNet " << state.netName(p.first) << " declared here";
  }
  state.sawError = true;
}

// Diagnose a `ttl.copy` whose endpoint is a pipe but whose enclosing domain
// extends outside the pipe's source/destination set.
void verifyCopy(CopyOp copyOp, const Domain &current, Operation *unanalyzable,
                ModuleState &state) {
  if (auto dstPipeType = mlir::dyn_cast<PipeType>(copyOp.getDst().getType())) {
    int64_t netId = dstPipeType.getPipeNetId();
    std::string name = state.netName(netId);
    std::string msg;
    llvm::raw_string_ostream(msg)
        << "this `ttl.copy(buffer, pipe)` sends data on PipeNet " << name
        << " from a node that is not a source of any pipe in that net; "
           "wrap the copy in `"
        << name << ".if_src(...)` or guard with `if " << name
        << ".is_src(): ...`";
    checkKnownSubset(copyOp, current, pipeSourceDomain(dstPipeType),
                     unanalyzable, msg, {{netId, PipeRole::Source}}, state);
    return;
  }
  if (auto srcPipeType = mlir::dyn_cast<PipeType>(copyOp.getSrc().getType())) {
    int64_t netId = srcPipeType.getPipeNetId();
    std::string name = state.netName(netId);
    std::string msg;
    llvm::raw_string_ostream(msg)
        << "this `ttl.copy(pipe, buffer)` receives data from PipeNet " << name
        << " on a node that is not a destination of any pipe in that "
           "net; wrap the copy in `"
        << name << ".if_dst(...)` or guard with `if " << name
        << ".is_dst(): ...`";
    checkKnownSubset(copyOp, current, pipeDestinationDomain(srcPipeType),
                     unanalyzable, msg, {{netId, PipeRole::Destination}},
                     state);
  }
}

// Decode the `pipenet_scope` (id, role) declarations into the union role
// domain plus the parallel role list used for diagnostic notes.
struct ScopeRoles {
  Domain domain;
  SmallVector<std::pair<int64_t, PipeRole>> roles;
};

// Read the (id, role) attribute pair from a `pipenet_scope` op and resolve
// it against `state`. Returns nullopt and emits an op error on a malformed
// scope.
std::optional<ScopeRoles> getPipeNetScopeRoles(PipeNetScopeOp scopeOp,
                                               ModuleState &state) {
  SmallVector<int64_t> ids;
  SmallVector<int64_t> roles;
  if (!readI64Array(scopeOp.getOperation(), kPipeNetIdsAttrName, ids) ||
      !readI64Array(scopeOp.getOperation(), kPipeNetRolesAttrName, roles)) {
    scopeOp.emitOpError() << "requires `" << kPipeNetIdsAttrName << "` and `"
                          << kPipeNetRolesAttrName << "` attributes";
    state.sawError = true;
    return std::nullopt;
  }
  if (ids.size() != roles.size()) {
    scopeOp.emitOpError() << "requires equal-length PipeNet id and role arrays";
    state.sawError = true;
    return std::nullopt;
  }
  ScopeRoles result;
  for (auto [pipeNetId, roleValue] : llvm::zip_equal(ids, roles)) {
    if (roleValue != static_cast<int64_t>(PipeRole::Source) &&
        roleValue != static_cast<int64_t>(PipeRole::Destination)) {
      scopeOp.emitOpError() << "has invalid PipeNet role " << roleValue
                            << " (expected 0=src or 1=dst)";
      state.sawError = true;
      return std::nullopt;
    }
    auto role = static_cast<PipeRole>(roleValue);
    Domain roleDomain = state.getRoleDomain(pipeNetId, role);
    result.domain = domainUnion(result.domain, roleDomain);
    result.roles.emplace_back(pipeNetId, role);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Lattice and analysis.
//===----------------------------------------------------------------------===//

// Lattice element for the guard analysis, attached to each program point.
// Stores two things:
//   - `domain_`: the launch coords that can reach this point under the
//     enclosing region predicates.
//   - `unanalyzableOp_`: the predicate op the verifier could not statically
//     evaluate, used to attach a note on downstream pipe-coupled diagnostics.
// `join` is set-union on `domain_` and `pickEarlierBySourceLoc` on
// `unanalyzableOp_`, applied at control-flow merges.
class DomainLattice : public dataflow::AbstractDenseLattice {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DomainLattice)

  using AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult join(const AbstractDenseLattice &rhs) override {
    const auto &other = static_cast<const DomainLattice &>(rhs);
    Domain joined = domainUnion(domain_, other.domain_);
    Operation *carriedOp =
        pickEarlierBySourceLoc(unanalyzableOp_, other.unanalyzableOp_);
    if (joined == domain_ && carriedOp == unanalyzableOp_) {
      return ChangeResult::NoChange;
    }
    domain_ = std::move(joined);
    unanalyzableOp_ = carriedOp;
    return ChangeResult::Change;
  }

  ChangeResult setDomain(Domain d, Operation *unanalyzable = nullptr) {
    if (d == domain_ && unanalyzable == unanalyzableOp_) {
      return ChangeResult::NoChange;
    }
    domain_ = std::move(d);
    unanalyzableOp_ = unanalyzable;
    return ChangeResult::Change;
  }

  void print(raw_ostream &os) const override {
    if (!domain_.known) {
      os << "<unknown>";
      return;
    }
    os << "{";
    bool first = true;
    for (Coord c : domain_.nodes) {
      if (!first) {
        os << ", ";
      }
      first = false;
      os << "(" << c.x << "," << c.y << ")";
    }
    os << "}";
  }

  const Domain &getDomain() const { return domain_; }
  Operation *getUnanalyzableOp() const { return unanalyzableOp_; }

private:
  Domain domain_;
  // When `domain_.known` is false, points at the predicate op the verifier
  // could not statically evaluate, so a downstream pipe-coupled op's
  // diagnostic can attach a note at that location.
  Operation *unanalyzableOp_ = nullptr;
};

// Forward dense dataflow analysis that propagates a `DomainLattice` through
// the IR and runs the verifier checks at each operation.
//
// At entry to a kernel-thread function the lattice is `baseDomain` (the full
// launch grid). The analysis narrows the lattice in three places:
//   - Region entry (`visitRegionBranchControlFlowTransfer`): an `scf.if` /
//     `affine.if` / `ttl.if_src` / `ttl.if_dst` / `ttl.pipenet_scope` shrinks
//     the inherited domain to the subset of coords where its predicate holds.
//   - Operation visit (`visitOperation`): pipe-coupled ops (`ttl.copy` on a
//     pipe) are checked against the current narrowed domain. CB push/wait ops
//     are recorded into `ModuleState` for the post-pass `verifyCBWaits`
//     cross-check.
//   - Control-flow merges: handled by `DomainLattice::join` (set-union meet).
class GuardAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<DomainLattice> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GuardAnalysis)

  GuardAnalysis(DataFlowSolver &solver, ModuleState &state)
      : DenseForwardDataFlowAnalysis(solver), state(state) {}

  void setToEntryState(DomainLattice *lattice) override {
    propagateIfChanged(lattice, lattice->setDomain(state.baseDomain));
  }

  LogicalResult visitOperation(Operation *op, const DomainLattice &before,
                               DomainLattice *after) override {
    ChangeResult result = after->join(before);

    TypeSwitch<Operation *>(op)
        .Case<CopyOp>([&](CopyOp copy) {
          verifyCopy(copy, before.getDomain(), before.getUnanalyzableOp(),
                     state);
          state.recordPipeEvent(copy, before.getDomain());
        })
        .Case<WaitOp>([&](WaitOp wait) {
          state.recordPipeWaitEvent(wait, before.getDomain(),
                                    before.getUnanalyzableOp());
        })
        .Case<CBPushOp>([&](CBPushOp push) {
          if (auto cbIndex = getCBIndex(push.getCb())) {
            state.cbProducerDomains[*cbIndex] = domainUnion(
                state.cbProducerDomains[*cbIndex], before.getDomain());
          }
        })
        .Case<CBWaitOp>([&](CBWaitOp wait) {
          if (auto cbIndex = getCBIndex(wait.getCb())) {
            state.waitUses.push_back({wait, before.getDomain(), *cbIndex});
          }
        });

    propagateIfChanged(after, result);
    return success();
  }

  // Compute the lattice at the entry of a region inside a
  // `RegionBranchOpInterface` op. The parent-into-region case
  // (`regionFrom == nullopt`) narrows `before.getDomain()` to the coords
  // where the chosen region's predicate holds; region exit and
  // sibling-to-sibling transitions delegate to the base implementation.
  void visitRegionBranchControlFlowTransfer(RegionBranchOpInterface branch,
                                            std::optional<unsigned> regionFrom,
                                            std::optional<unsigned> regionTo,
                                            const DomainLattice &before,
                                            DomainLattice *after) override {
    auto defaultHandling = [&]() {
      AbstractDenseForwardDataFlowAnalysis::
          visitRegionBranchControlFlowTransfer(branch, regionFrom, regionTo,
                                               before, after);
    };
    if (!regionTo || regionFrom) {
      // Exit to parent or transition between sibling regions: pass through.
      defaultHandling();
      return;
    }

    Operation *op = branch.getOperation();
    Domain narrowed = before.getDomain();
    Operation *unanalyzableOp = before.getUnanalyzableOp();

    TypeSwitch<Operation *>(op)
        .Case<scf::IfOp>([&](scf::IfOp ifOp) {
          BranchDomains domains =
              getBranchDomains(ifOp.getCondition(), before.getDomain(), state);
          unanalyzableOp = domains.unanalyzableOp;
          narrowed = (*regionTo == 0) ? domains.thenDomain : domains.elseDomain;
        })
        .Case<affine::AffineIfOp>([&](affine::AffineIfOp ifOp) {
          DomainResult condDomain = getAffineIfDomain(ifOp, state.baseDomain);
          unanalyzableOp = condDomain.unanalyzableOp;
          if (!condDomain.domain.known) {
            narrowed = Domain::unknown();
          } else if (*regionTo == 0) {
            narrowed = domainIntersect(before.getDomain(), condDomain.domain);
          } else {
            narrowed = domainIntersect(
                before.getDomain(),
                domainSubtract(state.baseDomain, condDomain.domain));
          }
        })
        .Case<IfSrcOp>([&](IfSrcOp ifSrc) {
          auto pipeType = mlir::cast<PipeType>(ifSrc.getPipe().getType());
          narrowed =
              domainIntersect(before.getDomain(), pipeSourceDomain(pipeType));
        })
        .Case<IfDstOp>([&](IfDstOp ifDst) {
          auto pipeType = mlir::cast<PipeType>(ifDst.getPipe().getType());
          narrowed = domainIntersect(before.getDomain(),
                                     pipeDestinationDomain(pipeType));
        })
        .Case<PipeNetScopeOp>([&](PipeNetScopeOp scopeOp) {
          auto scope = getPipeNetScopeRoles(scopeOp, state);
          if (!scope) {
            return;
          }
          std::string msg;
          {
            llvm::raw_string_ostream os(msg);
            SmallVector<int64_t> uniqueIds;
            for (auto &p : scope->roles) {
              if (!llvm::is_contained(uniqueIds, p.first)) {
                uniqueIds.push_back(p.first);
              }
            }
            os << "this region exchanges data on PipeNet";
            if (uniqueIds.size() != 1) {
              os << "s";
            }
            os << " ";
            llvm::interleaveComma(uniqueIds, os,
                                  [&](int64_t id) { os << state.netName(id); });
            os << " on launched nodes that are not part of "
               << (uniqueIds.size() == 1 ? "that net" : "those nets")
               << "; wrap the surrounding work in `if "
               << formatGuardExpression(scope->roles, state)
               << ": ...` so non-participating nodes skip it";
          }
          checkKnownSubset(scopeOp, before.getDomain(), scope->domain,
                           /*unanalyzableOp=*/nullptr, msg, scope->roles,
                           state);
          // Body domain is unchanged; the scope is a marker, not a
          // narrowing predicate.
        })
        .Default([&](auto) {});

    ChangeResult result = after->setDomain(narrowed, unanalyzableOp);
    propagateIfChanged(after, result);
  }

private:
  ModuleState &state;
};

// Cross-check each recorded `cb_wait` against the producer domain collected
// for the same dataflow buffer. Errors when the wait's lattice domain is not
// covered by any producer (deadlock-prone IR).
void verifyCBWaits(ModuleState &state) {
  for (WaitUse &use : state.waitUses) {
    auto it = state.cbProducerDomains.find(use.cbIndex);
    if (it == state.cbProducerDomains.end()) {
      use.op.emitOpError()
          << "this `cb_wait` reads from a dataflow buffer that no other "
             "thread fills; check that another `@ttl.compute()` or "
             "`@ttl.datamovement()` thread reserves and pushes the same "
             "buffer";
      state.sawError = true;
      continue;
    }
    checkKnownSubset(use.op, use.domain, it->second,
                     /*unanalyzableOp=*/nullptr,
                     "this `cb_wait` runs on launched nodes where no "
                     "thread pushes data to the buffer (would deadlock); "
                     "guard the wait with the same `if net.is_active(): "
                     "...` predicate the producer uses",
                     /*roles=*/{}, state);
  }
}

enum class PipeScheduleNodeKind { Send, ReceivePost, ReceiveWait };
enum class PipeScheduleEdgeKind {
  ProgramOrder,
  ReceivePostEnablesSend,
  SendCompletesReceive
};

struct PipeScheduleEdge {
  unsigned successor;
  PipeScheduleEdgeKind kind;
};

struct PipeScheduleNode {
  Operation *op;
  PipeType pipeType;
  Coord coord;
  PipeScheduleNodeKind kind;
  SmallVector<PipeScheduleEdge> successors;
};

using PipeIdentity =
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
using PipeCoordIdentity =
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
               int64_t, int64_t>;
using PipeNodeIdentity = std::tuple<Operation *, int64_t, int64_t, int64_t>;
using ProgramPointIdentity = std::tuple<Operation *, int64_t, int64_t>;

PipeIdentity getPipeIdentity(PipeType pipeType) {
  return {pipeType.getPipeNetId(), pipeType.getSrcX(),
          pipeType.getSrcY(),      pipeType.getDstStartX(),
          pipeType.getDstEndX(),   pipeType.getDstStartY(),
          pipeType.getDstEndY()};
}

PipeCoordIdentity getPipeCoordIdentity(PipeType pipeType, Coord coord) {
  auto [pipeNetId, srcX, srcY, dstStartX, dstEndX, dstStartY, dstEndY] =
      getPipeIdentity(pipeType);
  return {pipeNetId, srcX,    srcY,    dstStartX, dstEndX,
          dstStartY, dstEndY, coord.x, coord.y};
}

PipeNodeIdentity getPipeNodeIdentity(Operation *op, Coord coord,
                                     PipeScheduleNodeKind kind) {
  return {op, coord.x, coord.y, static_cast<int64_t>(kind)};
}

unsigned
addPipeScheduleNode(SmallVectorImpl<PipeScheduleNode> &nodes,
                    llvm::DenseMap<PipeNodeIdentity, unsigned> &nodeIds,
                    Operation *op, PipeType pipeType, Coord coord,
                    PipeScheduleNodeKind kind) {
  PipeNodeIdentity identity = getPipeNodeIdentity(op, coord, kind);
  auto [it, inserted] = nodeIds.try_emplace(identity, nodes.size());
  if (inserted) {
    nodes.push_back({op, pipeType, coord, kind, {}});
  }
  return it->second;
}

void addPipeScheduleEdge(SmallVectorImpl<PipeScheduleNode> &nodes,
                         unsigned predecessor, unsigned successor,
                         PipeScheduleEdgeKind kind) {
  SmallVectorImpl<PipeScheduleEdge> &successors = nodes[predecessor].successors;
  if (!llvm::any_of(successors, [&](const PipeScheduleEdge &edge) {
        return edge.successor == successor && edge.kind == kind;
      })) {
    successors.push_back({successor, kind});
  }
}

std::optional<SmallVector<unsigned>>
findPipeScheduleCycle(ArrayRef<PipeScheduleNode> nodes) {
  SmallVector<unsigned> stack;
  SmallVector<unsigned> cycle;
  SmallVector<uint8_t> colors(nodes.size(), 0);

  std::function<bool(unsigned)> visit = [&](unsigned nodeId) {
    colors[nodeId] = 1;
    stack.push_back(nodeId);
    for (const PipeScheduleEdge &edge : nodes[nodeId].successors) {
      unsigned successor = edge.successor;
      if (colors[successor] == 0) {
        if (visit(successor)) {
          return true;
        }
        continue;
      }
      if (colors[successor] != 1) {
        continue;
      }
      auto cycleStart = llvm::find(stack, successor);
      cycle.append(cycleStart, stack.end());
      cycle.push_back(successor);
      return true;
    }
    stack.pop_back();
    colors[nodeId] = 2;
    return false;
  };

  for (unsigned nodeId = 0, count = nodes.size(); nodeId < count; ++nodeId) {
    if (colors[nodeId] == 0 && visit(nodeId)) {
      return cycle;
    }
  }
  return std::nullopt;
}

std::optional<PipeScheduleEdgeKind>
getPipeScheduleEdgeKind(ArrayRef<PipeScheduleNode> nodes, unsigned predecessor,
                        unsigned successor) {
  for (const PipeScheduleEdge &edge : nodes[predecessor].successors) {
    if (edge.successor == successor) {
      return edge.kind;
    }
  }
  return std::nullopt;
}

bool cycleContainsEdge(ArrayRef<PipeScheduleNode> nodes,
                       ArrayRef<unsigned> cycle, unsigned predecessor,
                       unsigned successor, PipeScheduleEdgeKind kind) {
  for (unsigned idx = 0, count = cycle.size() - 1; idx < count; ++idx) {
    if (cycle[idx] != predecessor || cycle[idx + 1] != successor) {
      continue;
    }
    std::optional<PipeScheduleEdgeKind> actualKind =
        getPipeScheduleEdgeKind(nodes, predecessor, successor);
    if (actualKind && *actualKind == kind) {
      return true;
    }
  }
  return false;
}

bool cycleHasProgramOrderPath(ArrayRef<PipeScheduleNode> nodes,
                              ArrayRef<unsigned> cycle,
                              unsigned startCycleIndex,
                              unsigned endCycleIndex) {
  assert(startCycleIndex < endCycleIndex &&
         "expected a forward range within the reported cycle");
  for (unsigned idx = startCycleIndex; idx < endCycleIndex; ++idx) {
    std::optional<PipeScheduleEdgeKind> edgeKind =
        getPipeScheduleEdgeKind(nodes, cycle[idx], cycle[idx + 1]);
    if (!edgeKind || *edgeKind != PipeScheduleEdgeKind::ProgramOrder) {
      return false;
    }
  }
  return true;
}

std::string describePipeScheduleNode(const PipeScheduleNode &node) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  switch (node.kind) {
  case PipeScheduleNodeKind::Send:
    os << "send";
    break;
  case PipeScheduleNodeKind::ReceivePost:
    os << "destination address publication";
    break;
  case PipeScheduleNodeKind::ReceiveWait:
    os << "receive completion";
    break;
  }
  os << " at core_x=" << node.coord.x << ", core_y=" << node.coord.y;
  return buffer;
}

std::string describePipeScheduleEdge(const PipeScheduleNode &predecessor,
                                     const PipeScheduleNode &successor,
                                     PipeScheduleEdgeKind kind) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  switch (kind) {
  case PipeScheduleEdgeKind::ProgramOrder:
    os << "program order requires " << describePipeScheduleNode(successor)
       << " after " << describePipeScheduleNode(predecessor);
    break;
  case PipeScheduleEdgeKind::ReceivePostEnablesSend:
    os << "sender waits for " << describePipeScheduleNode(predecessor)
       << " before " << describePipeScheduleNode(successor);
    break;
  case PipeScheduleEdgeKind::SendCompletesReceive:
    os << describePipeScheduleNode(successor) << " waits for "
       << describePipeScheduleNode(predecessor) << " to transfer data";
    break;
  }
  return buffer;
}

std::optional<std::pair<unsigned, unsigned>>
findReceiveWaitBeforeCompletingSend(ArrayRef<PipeScheduleNode> nodes,
                                    ArrayRef<unsigned> cycle) {
  unsigned cycleNodeCount = cycle.size() - 1;
  for (unsigned waitIdx = 0; waitIdx < cycleNodeCount; ++waitIdx) {
    unsigned waitNodeId = cycle[waitIdx];
    const PipeScheduleNode &waitNode = nodes[waitNodeId];
    if (waitNode.kind != PipeScheduleNodeKind::ReceiveWait) {
      continue;
    }
    for (unsigned sendIdx = waitIdx + 1; sendIdx < cycle.size(); ++sendIdx) {
      unsigned sendNodeId = cycle[sendIdx];
      const PipeScheduleNode &sendNode = nodes[sendNodeId];
      if (sendNode.kind != PipeScheduleNodeKind::Send) {
        continue;
      }
      if (!cycleHasProgramOrderPath(nodes, cycle, waitIdx, sendIdx)) {
        continue;
      }
      if (cycleContainsEdge(nodes, cycle, sendNodeId, waitNodeId,
                            PipeScheduleEdgeKind::SendCompletesReceive)) {
        return std::make_pair(waitNodeId, sendNodeId);
      }
    }
  }
  return std::nullopt;
}

std::optional<std::pair<unsigned, unsigned>>
findSendBeforeReceivePost(ArrayRef<PipeScheduleNode> nodes,
                          ArrayRef<unsigned> cycle) {
  unsigned cycleNodeCount = cycle.size() - 1;
  for (unsigned sendIdx = 0; sendIdx < cycleNodeCount; ++sendIdx) {
    unsigned sendNodeId = cycle[sendIdx];
    const PipeScheduleNode &sendNode = nodes[sendNodeId];
    if (sendNode.kind != PipeScheduleNodeKind::Send) {
      continue;
    }
    for (unsigned postIdx = sendIdx + 1; postIdx < cycle.size(); ++postIdx) {
      unsigned postNodeId = cycle[postIdx];
      const PipeScheduleNode &postNode = nodes[postNodeId];
      if (postNode.kind != PipeScheduleNodeKind::ReceivePost) {
        continue;
      }
      if (!cycleHasProgramOrderPath(nodes, cycle, sendIdx, postIdx)) {
        continue;
      }
      if (cycleContainsEdge(nodes, cycle, postNodeId, sendNodeId,
                            PipeScheduleEdgeKind::ReceivePostEnablesSend)) {
        return std::make_pair(sendNodeId, postNodeId);
      }
    }
  }
  return std::nullopt;
}

void emitPipeScheduleCycleNotes(InFlightDiagnostic &diag,
                                ArrayRef<PipeScheduleNode> nodes,
                                ArrayRef<unsigned> cycle) {
  for (unsigned idx = 0, count = cycle.size() - 1; idx < count; ++idx) {
    unsigned predecessorId = cycle[idx];
    unsigned successorId = cycle[idx + 1];
    std::optional<PipeScheduleEdgeKind> edgeKind =
        getPipeScheduleEdgeKind(nodes, predecessorId, successorId);
    if (!edgeKind) {
      continue;
    }
    const PipeScheduleNode &predecessor = nodes[predecessorId];
    const PipeScheduleNode &successor = nodes[successorId];
    diag.attachNote(successor.op->getLoc())
        << describePipeScheduleEdge(predecessor, successor, *edgeKind);
    if (idx + 1 >= kMaxPipeScheduleCycleNotes) {
      break;
    }
  }
}

void emitPipeScheduleCycleDiagnostic(ArrayRef<PipeScheduleNode> nodes,
                                     ArrayRef<unsigned> cycle,
                                     ModuleState &state) {
  if (auto waitBeforeSend = findReceiveWaitBeforeCompletingSend(nodes, cycle)) {
    const PipeScheduleNode &waitNode = nodes[waitBeforeSend->first];
    const PipeScheduleNode &sendNode = nodes[waitBeforeSend->second];
    auto diag = waitNode.op->emitOpError()
                << "receive wait occurs before the send that completes it on "
                   "PipeNet "
                << state.netName(waitNode.pipeType.getPipeNetId());
    diag.attachNote(waitNode.op->getLoc())
        << "this wait blocks until the sender transfers into the posted "
           "destination dataflow buffer slot";
    diag.attachNote(sendNode.op->getLoc())
        << "this send is ordered after the wait in the same data-movement "
           "thread";
    diag.attachNote(waitNode.op->getLoc())
        << "move the receive wait after the send, or place send and receive in "
           "separate data-movement threads";
    state.sawError = true;
    return;
  }

  if (auto sendBeforePost = findSendBeforeReceivePost(nodes, cycle)) {
    const PipeScheduleNode &sendNode = nodes[sendBeforePost->first];
    const PipeScheduleNode &postNode = nodes[sendBeforePost->second];
    auto diag = sendNode.op->emitOpError()
                << "pipe send occurs before the receiver publishes a "
                   "destination address on PipeNet "
                << state.netName(sendNode.pipeType.getPipeNetId());
    diag.attachNote(sendNode.op->getLoc())
        << "this send waits for each destination to execute "
           "`ttl.copy(pipe, dst)`";
    diag.attachNote(postNode.op->getLoc())
        << "this destination address publication is ordered after the send in "
           "the "
           "same data-movement thread";
    diag.attachNote(sendNode.op->getLoc())
        << "move `ttl.copy(pipe, dst)` before the dependent send, or place "
           "send "
           "and receive in separate data-movement threads";
    state.sawError = true;
    return;
  }

  PipeScheduleNode node = nodes[cycle.front()];
  auto diag = node.op->emitOpError()
              << "pipe schedule contains a wait-for cycle on PipeNet "
              << state.netName(node.pipeType.getPipeNetId())
              << "; post the receive before the dependent send, or place the "
                 "send and receive in separate data-movement threads";

  emitPipeScheduleCycleNotes(diag, nodes, cycle);
  state.sawError = true;
}

// Verify the hidden pipe synchronization introduced by receiver-advertised pipe
// lowering. Receive-side ttl.copy publishes the address; ttl.wait on that
// handle waits for completion. Modeling those as distinct events preserves
// async copy semantics while rejecting wait-for cycles.
void verifyPipeScheduleCycles(ModuleOp module, ModuleState &state) {
  SmallVector<PipeScheduleNode> nodes;
  SmallVector<std::pair<unsigned, PipeType>> sendNodes;
  llvm::DenseMap<PipeNodeIdentity, unsigned> nodeIds;
  llvm::DenseMap<PipeCoordIdentity, SmallVector<unsigned>> receivePostNodes;
  llvm::DenseMap<PipeCoordIdentity, SmallVector<unsigned>> receiveWaitNodes;
  llvm::DenseMap<ProgramPointIdentity, unsigned> lastCompletionNodes;

  module.walk([&](Operation *op) {
    auto eventIt = state.pipeEventIndices.find(op);
    if (eventIt == state.pipeEventIndices.end()) {
      return;
    }
    PipeEvent event = state.pipeEvents[eventIt->second];
    if (!event.domain.known || event.domain.nodes.empty()) {
      return;
    }

    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp) {
      return;
    }

    for (Coord coord : event.domain.nodes) {
      PipeScheduleNodeKind nodeKind;
      if (event.kind == PipeEventKind::Send) {
        nodeKind = PipeScheduleNodeKind::Send;
      } else if (event.kind == PipeEventKind::ReceivePost) {
        nodeKind = PipeScheduleNodeKind::ReceivePost;
      } else {
        nodeKind = PipeScheduleNodeKind::ReceiveWait;
      }

      unsigned nodeId = addPipeScheduleNode(nodes, nodeIds, op, event.pipeType,
                                            coord, nodeKind);

      if (event.kind == PipeEventKind::Send) {
        sendNodes.push_back({nodeId, event.pipeType});
      } else if (event.kind == PipeEventKind::ReceivePost) {
        receivePostNodes[getPipeCoordIdentity(event.pipeType, coord)].push_back(
            nodeId);
      } else {
        receiveWaitNodes[getPipeCoordIdentity(event.pipeType, coord)].push_back(
            nodeId);
      }

      ProgramPointIdentity programPoint{funcOp.getOperation(), coord.x,
                                        coord.y};
      auto lastIt = lastCompletionNodes.find(programPoint);
      if (lastIt != lastCompletionNodes.end()) {
        addPipeScheduleEdge(nodes, lastIt->second, nodeId,
                            PipeScheduleEdgeKind::ProgramOrder);
      }
      lastCompletionNodes[programPoint] = nodeId;
    }
  });

  for (auto [sendNode, pipeType] : sendNodes) {
    Domain destinations = pipeDestinationDomain(pipeType);
    for (Coord coord : destinations.nodes) {
      PipeCoordIdentity identity = getPipeCoordIdentity(pipeType, coord);
      auto postIt = receivePostNodes.find(identity);
      if (postIt != receivePostNodes.end()) {
        for (unsigned receivePostNode : postIt->second) {
          addPipeScheduleEdge(nodes, receivePostNode, sendNode,
                              PipeScheduleEdgeKind::ReceivePostEnablesSend);
        }
      }
      auto waitIt = receiveWaitNodes.find(identity);
      if (waitIt != receiveWaitNodes.end()) {
        for (unsigned receiveWaitNode : waitIt->second) {
          addPipeScheduleEdge(nodes, sendNode, receiveWaitNode,
                              PipeScheduleEdgeKind::SendCompletesReceive);
        }
      }
    }
  }

  if (std::optional<SmallVector<unsigned>> cycle =
          findPipeScheduleCycle(nodes)) {
    emitPipeScheduleCycleDiagnostic(nodes, *cycle, state);
  }
}

// Walk the module and report any `pipenet_scope` or PipeNetPredicate that
// references a PipeNet id not declared by some `ttl.create_pipe`.
void validatePipeNetReferences(ModuleOp module, ModuleState &state) {
  module.walk([&](Operation *op) {
    auto report = [&](int64_t netId) {
      op->emitOpError() << "references unknown PipeNet " << state.netName(netId)
                        << " (id " << netId
                        << "); no `ttl.create_pipe` declares this net";
      state.sawError = true;
    };
    if (auto pred = mlir::dyn_cast<PipeNetPredicateOpInterface>(op)) {
      if (!state.pipeNetLocs.count(pred.getReferencedPipeNetId())) {
        report(pred.getReferencedPipeNetId());
      }
      return;
    }
    if (mlir::isa<PipeNetScopeOp>(op)) {
      SmallVector<int64_t> ids;
      if (readI64Array(op, kPipeNetIdsAttrName, ids)) {
        for (int64_t id : ids) {
          if (!state.pipeNetLocs.count(id)) {
            report(id);
          }
        }
      }
    }
  });
}

struct TTLVerifyPipeNetGuardsPass
    : impl::TTLVerifyPipeNetGuardsBase<TTLVerifyPipeNetGuardsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    ModuleState state;
    if (failed(state.initialize(module))) {
      signalPassFailure();
      return;
    }
    if (!state.hasPipes()) {
      return;
    }

    validatePipeNetReferences(module, state);
    if (state.sawError) {
      signalPassFailure();
      return;
    }

    // Kernel-thread `func.func`s are runtime-invoked entry points with no
    // callers (so they are analysis roots and get `setToEntryState`); helpers
    // they call have the caller's narrowed lattice flow through `func.call`.
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<GuardAnalysis>(state);
    if (failed(solver.initializeAndRun(module))) {
      signalPassFailure();
      return;
    }

    verifyCBWaits(state);
    if (!state.sawError) {
      verifyPipeScheduleCycles(module, state);
    }

    if (state.sawError) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
