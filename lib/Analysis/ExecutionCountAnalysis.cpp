// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Analysis/ExecutionCountAnalysis.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/GenericDomTreeConstruction.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>

namespace mlir::tt::execution_count_detail {

struct BlockFlowGraph;

/// A block and its possible control-flow edges for one induction environment.
struct BlockFlowNode {
  BlockFlowGraph *parent = nullptr;
  Block *block = nullptr;
  SmallVector<BlockFlowNode *> successors;
  SmallVector<BlockFlowNode *> predecessors;

  /// Supply the parent required by LLVM's generic dominator tree.
  BlockFlowGraph *getParent() const { return parent; }

  /// Print the corresponding MLIR block in LLVM graph diagnostics.
  void printAsOperand(llvm::raw_ostream &output, bool printType) const {
    block->printAsOperand(output, printType);
  }
};

/// The possible block CFG for one region and induction environment.
struct BlockFlowGraph {
  using NodeList = SmallVector<BlockFlowNode>;

  NodeList nodes;
  BlockFlowNode *entry = nullptr;

  /// Supply the entry required by LLVM's generic dominator tree.
  BlockFlowNode &front() { return *entry; }
};

} // namespace mlir::tt::execution_count_detail

namespace llvm {

/// Restrict LLVM graph traversal to successors possible in this context.
template <>
struct GraphTraits<mlir::tt::execution_count_detail::BlockFlowNode *> {
  using NodeRef = mlir::tt::execution_count_detail::BlockFlowNode *;
  using ChildIteratorType = SmallVector<NodeRef>::const_iterator;

  static NodeRef getEntryNode(NodeRef node) { return node; }
  static ChildIteratorType child_begin(NodeRef node) {
    return node->successors.begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return node->successors.end();
  }
};

/// Provide the reverse contextual edges required by post-dominance.
template <>
struct GraphTraits<Inverse<mlir::tt::execution_count_detail::BlockFlowNode *>> {
  using NodeRef = mlir::tt::execution_count_detail::BlockFlowNode *;
  using ChildIteratorType = SmallVector<NodeRef>::const_iterator;

  static NodeRef getEntryNode(Inverse<NodeRef> graph) { return graph.Graph; }
  static ChildIteratorType child_begin(NodeRef node) {
    return node->predecessors.begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return node->predecessors.end();
  }
};

/// Let post-dominance inspect every node, including disconnected blocks.
template <>
struct GraphTraits<mlir::tt::execution_count_detail::BlockFlowGraph *>
    : GraphTraits<mlir::tt::execution_count_detail::BlockFlowNode *> {
  using GraphType = mlir::tt::execution_count_detail::BlockFlowGraph *;
  using NodeRef = mlir::tt::execution_count_detail::BlockFlowNode *;
  using nodes_iterator = pointer_iterator<
      mlir::tt::execution_count_detail::BlockFlowGraph::NodeList::iterator>;

  static NodeRef getEntryNode(GraphType graph) { return graph->entry; }
  static nodes_iterator nodes_begin(GraphType graph) {
    return nodes_iterator(graph->nodes.begin());
  }
  static nodes_iterator nodes_end(GraphType graph) {
    return nodes_iterator(graph->nodes.end());
  }
};

} // namespace llvm

namespace mlir::tt {

namespace {

using execution_count_detail::BlockFlowGraph;
using execution_count_detail::BlockFlowNode;

/// One nesting level between the root region and the queried operation. The
/// block contains either the query or its next inner enclosing operation.
struct ControlFrame {
  Operation *parent = nullptr;
  Region *region = nullptr;
  Block *targetBlock = nullptr;
};

/// Record whether each successor edge is possible, ordered by block and
/// successor position.
using BlockFlowKey = SmallVector<std::uint8_t>;

// A proven reducible loop's induction variable, kept as operands since the trip count is re-evaluated per query.
struct NaturalLoopInfo {
  Value lowerBoundOperand;
  SmallVector<Value, 1> stepOperands; // one per backward edge, all must agree
  Value boundOperand;
  bool ascending = true; // true for positive step, false for negative
  bool isSigned = true;
};

// Exact block counts for one possible block CFG.
struct BlockCountResult {
  llvm::DenseMap<Block *, std::optional<std::uint64_t>> blockCounts; // overridden by the maps below when a block is in one
  llvm::DenseMap<Block *, NaturalLoopInfo> loopHeaders; // count is the loop's trip count
  llvm::DenseMap<Block *, Block *> loopBodyBlocks; // runs once per iteration of the named header
  llvm::DenseMap<Block *, SmallVector<Block *, 2>> postLoopDependencies; // "runs once" only if every named header's loop terminates
};

/// Bounds the loop iterations examined while proving one operation count.
class EnumerationBudget {
public:
  explicit EnumerationBudget(std::uint64_t remainingIterations)
      : remainingIterations(remainingIterations) {}

  /// Return whether the requested iterations fit without consuming them.
  bool canConsume(std::uint64_t iterationCount) const {
    return iterationCount <= remainingIterations;
  }

  /// Consume one iteration, or return false when no budget remains.
  bool tryConsume() {
    if (remainingIterations == 0) {
      return false;
    }
    --remainingIterations;
    return true;
  }

private:
  std::uint64_t remainingIterations;
};

/// Return whether the parent may transition between the two child regions.
bool isRegionReachable(RegionBranchOpInterface branch, Region &sourceRegion,
                       Region &targetRegion) {
  SmallVector<Region *> worklist{&sourceRegion};
  llvm::SmallPtrSet<Region *, 4> visited;
  visited.insert(&sourceRegion);
  while (!worklist.empty()) {
    Region *region = worklist.pop_back_val();
    SmallVector<RegionSuccessor> successors;
    branch.getSuccessorRegions(*region, successors);
    for (RegionSuccessor successor : successors) {
      Region *successorRegion = successor.getSuccessor();
      if (!successorRegion) {
        continue;
      }
      if (successorRegion == &targetRegion) {
        return true;
      }
      if (visited.insert(successorRegion).second) {
        worklist.push_back(successorRegion);
      }
    }
  }
  return false;
}

arith::CmpIPredicate invertCmpIPredicate(arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
    return arith::CmpIPredicate::ne;
  case arith::CmpIPredicate::ne:
    return arith::CmpIPredicate::eq;
  case arith::CmpIPredicate::slt:
    return arith::CmpIPredicate::sge;
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::sgt;
  case arith::CmpIPredicate::sgt:
    return arith::CmpIPredicate::sle;
  case arith::CmpIPredicate::sge:
    return arith::CmpIPredicate::slt;
  case arith::CmpIPredicate::ult:
    return arith::CmpIPredicate::uge;
  case arith::CmpIPredicate::ule:
    return arith::CmpIPredicate::ugt;
  case arith::CmpIPredicate::ugt:
    return arith::CmpIPredicate::ule;
  case arith::CmpIPredicate::uge:
    return arith::CmpIPredicate::ult;
  }
  llvm_unreachable("unhandled arith::CmpIPredicate");
}

arith::CmpIPredicate swapCmpIPredicate(arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
    return arith::CmpIPredicate::eq;
  case arith::CmpIPredicate::ne:
    return arith::CmpIPredicate::ne;
  case arith::CmpIPredicate::slt:
    return arith::CmpIPredicate::sgt;
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::sge;
  case arith::CmpIPredicate::sgt:
    return arith::CmpIPredicate::slt;
  case arith::CmpIPredicate::sge:
    return arith::CmpIPredicate::sle;
  case arith::CmpIPredicate::ult:
    return arith::CmpIPredicate::ugt;
  case arith::CmpIPredicate::ule:
    return arith::CmpIPredicate::uge;
  case arith::CmpIPredicate::ugt:
    return arith::CmpIPredicate::ult;
  case arith::CmpIPredicate::uge:
    return arith::CmpIPredicate::ule;
  }
  llvm_unreachable("unhandled arith::CmpIPredicate");
}

// Return whether `value` is defined outside `blocks`, i.e. loop-invariant for an SCC made of exactly those blocks.
bool isDefinedOutsideBlocks(Value value,
                            const llvm::DenseSet<Block *> &blocks) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    return !blocks.contains(argument.getOwner());
  }
  return !blocks.contains(value.getDefiningOp()->getBlock());
}

// Return the operand `source` passes to `target`'s `argIndex`-th block argument, if there is exactly one such edge.
std::optional<Value> getSuccessorOperand(Block *source, Block *target,
                                         unsigned argIndex) {
  Operation *terminator = source->getTerminator();
  auto branch = dyn_cast_or_null<BranchOpInterface>(terminator);
  if (!branch) {
    return std::nullopt;
  }
  std::optional<Value> result;
  unsigned successorIndex = 0;
  for (Block *successor : terminator->getSuccessors()) {
    if (successor == target) {
      if (result) {
        return std::nullopt;
      }
      SuccessorOperands operands = branch.getSuccessorOperands(successorIndex);
      if (argIndex >= operands.size()) {
        return std::nullopt;
      }
      Value operand = operands[argIndex];
      if (!operand) {
        return std::nullopt;
      }
      result = operand;
    }
    ++successorIndex;
  }
  return result;
}

// Classify reachability, cyclic taint, and post-dominance; `resolvedCycleNodes` are SCC nodes already proven exact elsewhere.
llvm::DenseMap<Block *, std::optional<std::uint64_t>>
classifyAcyclicCounts(BlockFlowGraph &graph,
                      const llvm::SmallPtrSetImpl<BlockFlowNode *>
                          &resolvedCycleNodes) {
  llvm::DenseMap<Block *, std::optional<std::uint64_t>> counts;

  llvm::DenseSet<BlockFlowNode *> reachableNodes;
  SmallVector<BlockFlowNode *> worklist{graph.entry};
  reachableNodes.insert(graph.entry);
  while (!worklist.empty()) {
    BlockFlowNode *node = worklist.pop_back_val();
    for (BlockFlowNode *successor : node->successors) {
      if (reachableNodes.insert(successor).second) {
        worklist.push_back(successor);
      }
    }
  }

  llvm::DenseSet<BlockFlowNode *> cycleAffectedNodes;
  for (auto sccIt = llvm::scc_begin(&graph); !sccIt.isAtEnd(); ++sccIt) {
    if (!sccIt.hasCycle()) {
      continue;
    }
    for (BlockFlowNode *node : *sccIt) {
      if (resolvedCycleNodes.contains(node)) {
        continue;
      }
      if (cycleAffectedNodes.insert(node).second) {
        worklist.push_back(node);
      }
    }
  }
  while (!worklist.empty()) {
    BlockFlowNode *node = worklist.pop_back_val();
    for (BlockFlowNode *successor : node->successors) {
      if (cycleAffectedNodes.insert(successor).second) {
        worklist.push_back(successor);
      }
    }
  }

  // A reachable block executes exactly once iff every continuation visits it.
  llvm::PostDomTreeBase<BlockFlowNode> postDominance;
  postDominance.recalculate(graph);
  for (BlockFlowNode &node : graph.nodes) {
    std::optional<std::uint64_t> maybeCount;
    if (!reachableNodes.contains(&node)) {
      maybeCount = 0;
    } else if (!cycleAffectedNodes.contains(&node) &&
               postDominance.dominates(&node, graph.entry)) {
      maybeCount = 1;
    }
    counts.try_emplace(node.block, maybeCount);
  }
  return counts;
}

// Classify each block's per-iteration count for one pass through the SCC, by dropping back edges and exit edges to get an acyclic subgraph.
llvm::DenseMap<Block *, std::optional<std::uint64_t>>
computePerIterationCounts(ArrayRef<BlockFlowNode *> sccNodes,
                          BlockFlowNode *header) {
  llvm::SmallPtrSet<BlockFlowNode *, 8> sccSet(sccNodes.begin(),
                                               sccNodes.end());
  BlockFlowGraph subGraph;
  subGraph.nodes.reserve(sccNodes.size());
  llvm::DenseMap<BlockFlowNode *, BlockFlowNode *> nodeMap;
  for (BlockFlowNode *original : sccNodes) {
    BlockFlowNode &copy = subGraph.nodes.emplace_back();
    copy.parent = &subGraph;
    copy.block = original->block;
    nodeMap.try_emplace(original, &copy);
  }
  subGraph.entry = nodeMap.lookup(header);

  for (BlockFlowNode *original : sccNodes) {
    BlockFlowNode *source = nodeMap.lookup(original);
    for (BlockFlowNode *originalSuccessor : original->successors) {
      if (originalSuccessor == header || !sccSet.contains(originalSuccessor)) {
        continue;
      }
      BlockFlowNode *target = nodeMap.lookup(originalSuccessor);
      if (!llvm::is_contained(source->successors, target)) {
        source->successors.push_back(target);
        target->predecessors.push_back(source);
      }
    }
  }

  llvm::SmallPtrSet<BlockFlowNode *, 1> noResolvedCycles;
  return classifyAcyclicCounts(subGraph, noResolvedCycles);
}

// Facts proven for a reducible cyclic SCC, ready to record in a BlockCountResult.
struct ProvenNaturalLoop {
  Block *header = nullptr;
  NaturalLoopInfo info;
  llvm::DenseMap<Block *, std::optional<std::uint64_t>> perIterationCounts; // header is always 1
};

// Try to prove a cyclic SCC is a reducible single-entry loop with a constant-step induction variable and a loop-invariant exit bound; returns nullopt if the shape isn't recognized.
std::optional<ProvenNaturalLoop>
tryProveNaturalLoop(ArrayRef<BlockFlowNode *> sccNodes) {
  llvm::SmallPtrSet<BlockFlowNode *, 8> sccSet(sccNodes.begin(),
                                               sccNodes.end());
  llvm::DenseSet<Block *> sccBlocks;
  for (BlockFlowNode *node : sccNodes) {
    sccBlocks.insert(node->block);
  }

  // A cyclic SCC entered through one block only is automatically reducible.
  BlockFlowNode *header = nullptr;
  for (BlockFlowNode *node : sccNodes) {
    bool hasOutsidePredecessor =
        llvm::any_of(node->predecessors, [&](BlockFlowNode *predecessor) {
          return !sccSet.contains(predecessor);
        });
    if (!hasOutsidePredecessor) {
      continue;
    }
    if (header && header != node) {
      return std::nullopt;
    }
    header = node;
  }
  if (!header) {
    return std::nullopt;
  }

  SmallVector<BlockFlowNode *> preheaders;
  for (BlockFlowNode *predecessor : header->predecessors) {
    if (!sccSet.contains(predecessor)) {
      preheaders.push_back(predecessor);
    }
  }
  if (preheaders.size() != 1) {
    return std::nullopt;
  }
  BlockFlowNode *preheader = preheaders.front();

  SmallVector<BlockFlowNode *> latches;
  for (BlockFlowNode *node : sccNodes) {
    if (llvm::is_contained(node->successors, header)) {
      latches.push_back(node);
    }
  }
  if (latches.empty()) {
    return std::nullopt;
  }

  BlockFlowNode *exitSource = nullptr;
  for (BlockFlowNode *node : sccNodes) {
    for (BlockFlowNode *successor : node->successors) {
      if (sccSet.contains(successor)) {
        continue;
      }
      if (exitSource && exitSource != node) {
        return std::nullopt;
      }
      exitSource = node;
    }
  }
  if (!exitSource) {
    return std::nullopt;
  }
  if (exitSource == header) {
    // Only safe when the header is the SCC's only block, else it could run one extra time compared to the body (a while loop).
    if (sccNodes.size() != 1) {
      return std::nullopt;
    }
  } else if (!llvm::is_contained(latches, exitSource)) {
    return std::nullopt;
  }

  auto condBr =
      dyn_cast_or_null<cf::CondBranchOp>(exitSource->block->getTerminator());
  if (!condBr) {
    return std::nullopt;
  }
  bool trueLeavesSCC = !sccBlocks.contains(condBr.getTrueDest());
  bool falseLeavesSCC = !sccBlocks.contains(condBr.getFalseDest());
  if (trueLeavesSCC == falseLeavesSCC) {
    return std::nullopt;
  }
  auto cmpOp = condBr.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!cmpOp) {
    return std::nullopt;
  }
  arith::CmpIPredicate predicate = cmpOp.getPredicate();
  if (trueLeavesSCC) {
    predicate = invertCmpIPredicate(predicate);
  }

  for (BlockArgument ivArg : header->block->getArguments()) {
    unsigned argIndex = ivArg.getArgNumber();

    std::optional<Value> lowerBoundOperand =
        getSuccessorOperand(preheader->block, header->block, argIndex);
    if (!lowerBoundOperand) {
      continue;
    }

    SmallVector<Value, 1> stepOperands;
    llvm::DenseMap<BlockFlowNode *, Value> latchNextValues;
    bool latchesMatch = true;
    for (BlockFlowNode *latch : latches) {
      std::optional<Value> latchOperand =
          getSuccessorOperand(latch->block, header->block, argIndex);
      if (!latchOperand) {
        latchesMatch = false;
        break;
      }
      auto addOp = latchOperand->getDefiningOp<arith::AddIOp>();
      if (!addOp) {
        latchesMatch = false;
        break;
      }
      Value step;
      if (addOp.getLhs() == ivArg) {
        step = addOp.getRhs();
      } else if (addOp.getRhs() == ivArg) {
        step = addOp.getLhs();
      } else {
        latchesMatch = false;
        break;
      }
      if (!isDefinedOutsideBlocks(step, sccBlocks)) {
        latchesMatch = false;
        break;
      }
      stepOperands.push_back(step);
      latchNextValues.try_emplace(latch, *latchOperand);
    }
    if (!latchesMatch) {
      continue;
    }

    Value lhs = cmpOp.getLhs();
    Value rhs = cmpOp.getRhs();
    // exitSource is always in latches, so this always resolves.
    Value testedNext = latchNextValues.lookup(exitSource);
    auto isTestedValue = [&](Value value) {
      return value == ivArg || (testedNext && value == testedNext);
    };
    bool testedIsLhs = isTestedValue(lhs);
    bool testedIsRhs = isTestedValue(rhs);
    if (testedIsLhs == testedIsRhs) {
      continue;
    }
    arith::CmpIPredicate normalizedPredicate =
        testedIsRhs ? swapCmpIPredicate(predicate) : predicate;
    Value boundOperand = testedIsLhs ? rhs : lhs;
    if (!isDefinedOutsideBlocks(boundOperand, sccBlocks)) {
      continue;
    }

    bool ascending;
    bool isSigned;
    switch (normalizedPredicate) {
    case arith::CmpIPredicate::slt:
      ascending = true;
      isSigned = true;
      break;
    case arith::CmpIPredicate::ult:
      ascending = true;
      isSigned = false;
      break;
    case arith::CmpIPredicate::sgt:
      ascending = false;
      isSigned = true;
      break;
    default:
      continue;
    }

    ProvenNaturalLoop proven;
    proven.header = header->block;
    proven.info = NaturalLoopInfo{*lowerBoundOperand, stepOperands,
                                  boundOperand, ascending, isSigned};
    proven.perIterationCounts = computePerIterationCounts(sccNodes, header);
    return proven;
  }
  return std::nullopt;
}

} // namespace

class ExecutionCountAnalysis::Impl {
public:
  Impl(Region &rootRegion, SymbolValueEvaluator symbolValueEvaluator,
       RegionInvocationCountEvaluator regionInvocationCountEvaluator,
       Options options)
      : rootRegion(rootRegion),
        symbolValueEvaluator(std::move(symbolValueEvaluator)),
        regionInvocationCountEvaluator(
            std::move(regionInvocationCountEvaluator)),
        options(options) {
    Operation *parent = rootRegion.getParentOp();
    if (!parent) {
      return;
    }
    dataFlowSolver = std::make_unique<DataFlowSolver>();
    dataflow::loadBaselineAnalyses(*dataFlowSolver);
    if (failed(dataFlowSolver->initializeAndRun(parent))) {
      dataFlowSolver.reset();
    }
  }

  std::optional<std::uint64_t> getExecutionCount(Operation *operation) {
    if (!operation) {
      return std::nullopt;
    }
    Block *operationBlock = operation->getBlock();
    if (!operationBlock) {
      return std::nullopt;
    }
    // Every operation in a block executes once per block invocation, so reuse
    // the control-flow proof for all operations in that block.
    auto cached = blockExecutionCountCache.find(operationBlock);
    if (cached != blockExecutionCountCache.end()) {
      return cached->second;
    }

    SmallVector<ControlFrame> frames;
    Block *rootBlock = nullptr;
    if (!collectControlFrames(operation, frames, rootBlock)) {
      blockExecutionCountCache.try_emplace(operationBlock, std::nullopt);
      return std::nullopt;
    }

    llvm::DenseMap<Value, llvm::APInt> inductionValues;
    EnumerationBudget enumerationBudget(options.maxEnumeratedIterations);
    std::optional<std::uint64_t> maybeRootBlockCount =
        getExactBlockInvocationCount(rootRegion, rootBlock, inductionValues);
    std::optional<std::uint64_t> maybeCount;
    if (!maybeRootBlockCount || *maybeRootBlockCount == 0) {
      maybeCount = maybeRootBlockCount;
    } else {
      std::optional<std::uint64_t> maybeNestedCount =
          countExecutions(frames, 0, inductionValues, enumerationBudget);
      maybeCount = maybeNestedCount
                       ? llvm::checkedMulUnsigned(*maybeRootBlockCount,
                                                  *maybeNestedCount)
                       : std::nullopt;
    }
    blockExecutionCountCache.try_emplace(operationBlock, maybeCount);
    return maybeCount;
  }

private:
  bool collectControlFrames(Operation *operation,
                            SmallVectorImpl<ControlFrame> &frames,
                            Block *&rootBlock) const {
    Region *parentRegion = operation->getParentRegion();
    if (!parentRegion || !rootRegion.isAncestor(parentRegion)) {
      return false;
    }

    for (Operation *child = operation;
         child->getParentRegion() != &rootRegion;) {
      Region *region = child->getParentRegion();
      Operation *parent = region->getParentOp();
      if (!parent) {
        return false;
      }
      frames.push_back({parent, region, child->getBlock()});
      child = parent;
    }
    Operation *rootChild = frames.empty() ? operation : frames.back().parent;
    rootBlock = rootChild->getBlock();
    if (!rootBlock || rootBlock->getParent() != &rootRegion) {
      return false;
    }
    std::reverse(frames.begin(), frames.end());
    return true;
  }

  IntegerExpressionEvaluator createIntegerEvaluator(
      const llvm::DenseMap<Value, llvm::APInt> &inductionValues) const {
    return IntegerExpressionEvaluator(
        [this, &inductionValues](Value value) -> std::optional<llvm::APInt> {
          if (auto inductionValue = inductionValues.find(value);
              inductionValue != inductionValues.end()) {
            return inductionValue->second;
          }
          if (std::optional<llvm::APInt> maybeConstant =
                  getDataFlowIntegerConstant(value)) {
            return maybeConstant;
          }
          return symbolValueEvaluator ? symbolValueEvaluator(value)
                                      : std::nullopt;
        });
  }

  /// Return an integer constant propagated through block and region arguments.
  std::optional<llvm::APInt> getDataFlowIntegerConstant(Value value) const {
    if (!dataFlowSolver) {
      return std::nullopt;
    }
    using ConstantLattice = dataflow::Lattice<dataflow::ConstantValue>;
    const auto *lattice = dataFlowSolver->lookupState<ConstantLattice>(value);
    if (!lattice || lattice->getValue().isUninitialized()) {
      return std::nullopt;
    }
    auto integerAttr =
        dyn_cast_or_null<IntegerAttr>(lattice->getValue().getConstantValue());
    return integerAttr ? std::optional(integerAttr.getValue()) : std::nullopt;
  }

  SmallVector<Attribute> evaluateOperands(
      Operation *operation,
      const llvm::DenseMap<Value, llvm::APInt> &inductionValues) const {
    SmallVector<Attribute> operands;
    operands.reserve(operation->getNumOperands());
    IntegerExpressionEvaluator integerEvaluator =
        createIntegerEvaluator(inductionValues);
    for (Value operand : operation->getOperands()) {
      std::optional<llvm::APInt> maybeInteger =
          operand.getType().isIntOrIndex() ? integerEvaluator.evaluate(operand)
                                           : std::nullopt;
      if (maybeInteger) {
        operands.push_back(IntegerAttr::get(operand.getType(), *maybeInteger));
        continue;
      }
      Attribute constant;
      operands.push_back(matchPattern(operand, m_Constant(&constant))
                             ? constant
                             : Attribute());
    }
    return operands;
  }

  bool isKnownDead(Block *block) const {
    if (!dataFlowSolver) {
      return false;
    }
    ProgramPoint *blockStart = dataFlowSolver->getProgramPointBefore(block);
    const auto *executable =
        dataFlowSolver->lookupState<dataflow::Executable>(blockStart);
    return executable && !executable->isLive();
  }

  bool isKnownDead(Block *source, Block *target) const {
    if (!dataFlowSolver) {
      return false;
    }
    dataflow::CFGEdge *edge =
        dataFlowSolver->getLatticeAnchor<dataflow::CFGEdge>(source, target);
    const auto *executable =
        dataFlowSolver->lookupState<dataflow::Executable>(edge);
    return executable && !executable->isLive();
  }

  /// Return the target block's exact count from its possible control-flow CFG.
  std::optional<std::uint64_t> getExactBlockInvocationCount(
      Region &region, Block *targetBlock,
      const llvm::DenseMap<Value, llvm::APInt> &inductionValues) {
    if (!targetBlock || targetBlock->getParent() != &region || region.empty()) {
      return std::nullopt;
    }

    FailureOr<BlockFlowKey> maybeFlowKey =
        getBlockFlowKey(region, inductionValues);
    if (failed(maybeFlowKey)) {
      return std::nullopt;
    }
    auto &regionCache = blockCountCache[&region];
    auto resultIt = regionCache.find(*maybeFlowKey);
    if (resultIt == regionCache.end()) {
      BlockCountResult result = classifyBlockCounts(region, *maybeFlowKey);
      auto [insertedIt, inserted] =
          regionCache.try_emplace(std::move(*maybeFlowKey), std::move(result));
      assert(inserted && "new block CFG must be inserted");
      resultIt = insertedIt;
    }

    BlockCountResult &result = resultIt->second;
    if (auto headerIt = result.loopHeaders.find(targetBlock);
        headerIt != result.loopHeaders.end()) {
      return evaluateNaturalLoopTripCount(headerIt->second, inductionValues);
    }
    if (auto bodyIt = result.loopBodyBlocks.find(targetBlock);
        bodyIt != result.loopBodyBlocks.end()) {
      auto headerIt = result.loopHeaders.find(bodyIt->second);
      assert(headerIt != result.loopHeaders.end() &&
             "loop body block must reference a proven header");
      return evaluateNaturalLoopTripCount(headerIt->second, inductionValues);
    }
    if (auto dependsIt = result.postLoopDependencies.find(targetBlock);
        dependsIt != result.postLoopDependencies.end()) {
      for (Block *header : dependsIt->second) {
        auto headerIt = result.loopHeaders.find(header);
        assert(headerIt != result.loopHeaders.end() &&
               "post-loop dependency must reference a proven header");
        if (!evaluateNaturalLoopTripCount(headerIt->second, inductionValues)) {
          return std::nullopt;
        }
      }
      return 1;
    }

    auto countIt = result.blockCounts.find(targetBlock);
    assert(countIt != result.blockCounts.end() &&
           "block-count result must contain every block");
    return countIt->second;
  }

  /// Encode the successor edges possible under the current induction values.
  FailureOr<BlockFlowKey> getBlockFlowKey(
      Region &region,
      const llvm::DenseMap<Value, llvm::APInt> &inductionValues) const {
    BlockFlowKey flowKey;
    for (Block &block : region) {
      Operation *terminator = block.getTerminator();
      if (!terminator) {
        return failure();
      }
      if (terminator->getNumSuccessors() == 0) {
        continue;
      }
      auto branch = dyn_cast<BranchOpInterface>(terminator);
      if (!branch) {
        return failure();
      }

      SmallVector<Attribute> operands =
          evaluateOperands(terminator, inductionValues);
      Block *selectedSuccessor = branch.getSuccessorForOperands(operands);
      if (selectedSuccessor &&
          !llvm::is_contained(terminator->getSuccessors(), selectedSuccessor)) {
        return failure();
      }

      for (Block *successor : terminator->getSuccessors()) {
        bool isPossible =
            !isKnownDead(&block) && !isKnownDead(successor) &&
            !isKnownDead(&block, successor) &&
            (!selectedSuccessor || successor == selectedSuccessor);
        flowKey.push_back(isPossible);
      }
    }
    return flowKey;
  }

  /// Classify every block in one invocation of the possible block CFG.
  BlockCountResult classifyBlockCounts(Region &region,
                                       ArrayRef<std::uint8_t> flowKey) const {
    BlockCountResult result;
    BlockFlowGraph graph;
    // Successor lists store node pointers, so allocate every node first.
    graph.nodes.reserve(std::distance(region.begin(), region.end()));
    llvm::DenseMap<Block *, BlockFlowNode *> nodesByBlock;
    for (Block &block : region) {
      BlockFlowNode &node = graph.nodes.emplace_back();
      node.parent = &graph;
      node.block = &block;
      nodesByBlock.try_emplace(&block, &node);
    }
    graph.entry = nodesByBlock.lookup(&region.front());

    std::size_t edgePosition = 0;
    for (Block &block : region) {
      BlockFlowNode *sourceNode = nodesByBlock.lookup(&block);
      Operation *terminator = block.getTerminator();
      for (Block *successor : terminator->getSuccessors()) {
        assert(edgePosition < flowKey.size() && "block-flow key is too short");
        if (!flowKey[edgePosition++]) {
          continue;
        }
        BlockFlowNode *successorNode = nodesByBlock.lookup(successor);
        assert(successorNode && "successor must belong to the same region");
        if (!llvm::is_contained(sourceNode->successors, successorNode)) {
          sourceNode->successors.push_back(successorNode);
          successorNode->predecessors.push_back(sourceNode);
        }
      }
    }
    assert(edgePosition == flowKey.size() && "block-flow key is too long");

    // Proven loops get an exact trip count; exclude their blocks from the cyclic-taint seeding below so downstream blocks aren't forced unknown.
    llvm::SmallPtrSet<BlockFlowNode *, 8> resolvedCycleNodes;
    SmallVector<std::pair<Block *, std::optional<std::uint64_t>>>
        forcedLoopBodyCounts;
    for (auto sccIt = llvm::scc_begin(&graph); !sccIt.isAtEnd(); ++sccIt) {
      if (!sccIt.hasCycle()) {
        continue;
      }
      SmallVector<BlockFlowNode *> sccNodes(sccIt->begin(), sccIt->end());
      std::optional<ProvenNaturalLoop> maybeProven =
          tryProveNaturalLoop(sccNodes);
      if (!maybeProven) {
        continue;
      }
      for (BlockFlowNode *node : sccNodes) {
        resolvedCycleNodes.insert(node);
      }
      Block *header = maybeProven->header;
      for (auto &[block, maybeMultiplier] : maybeProven->perIterationCounts) {
        if (block == header) {
          continue;
        }
        if (maybeMultiplier && *maybeMultiplier == 1) {
          result.loopBodyBlocks.try_emplace(block, header);
        } else {
          forcedLoopBodyCounts.emplace_back(block, maybeMultiplier);
        }
      }
      result.loopHeaders.try_emplace(header, std::move(maybeProven->info));
    }

    result.blockCounts = classifyAcyclicCounts(graph, resolvedCycleNodes);
    for (auto &[block, maybeMultiplier] : forcedLoopBodyCounts) {
      // Overrides the baseline classification, which can't see per-iteration counts inside a loop.
      result.blockCounts[block] = maybeMultiplier;
    }

    // A block after a loop that normally runs once is only guaranteed to run if the loop is proven to terminate.
    for (auto &headerEntry : result.loopHeaders) {
      Block *header = headerEntry.first;
      BlockFlowNode *headerNode = nodesByBlock.lookup(header);
      llvm::DenseSet<BlockFlowNode *> reachableFromHeader;
      SmallVector<BlockFlowNode *> headerWorklist{headerNode};
      while (!headerWorklist.empty()) {
        BlockFlowNode *node = headerWorklist.pop_back_val();
        for (BlockFlowNode *successor : node->successors) {
          if (reachableFromHeader.insert(successor).second) {
            headerWorklist.push_back(successor);
          }
        }
      }
      for (BlockFlowNode *node : reachableFromHeader) {
        if (node->block == header || result.loopBodyBlocks.count(node->block)) {
          continue;
        }
        auto countIt = result.blockCounts.find(node->block);
        if (countIt != result.blockCounts.end() && countIt->second &&
            *countIt->second == 1) {
          result.postLoopDependencies[node->block].push_back(header);
        }
      }
    }

    return result;
  }

  std::optional<std::uint64_t> getExactRegionInvocationCount(
      ControlFrame frame,
      const llvm::DenseMap<Value, llvm::APInt> &inductionValues) const {
    if (regionInvocationCountEvaluator) {
      if (std::optional<std::uint64_t> maybeCount =
              regionInvocationCountEvaluator(*frame.region)) {
        return maybeCount;
      }
    }

    auto branch = dyn_cast<RegionBranchOpInterface>(frame.parent);
    if (!branch) {
      return std::nullopt;
    }

    SmallVector<Attribute> operands =
        evaluateOperands(frame.parent, inductionValues);

    SmallVector<InvocationBounds> bounds;
    branch.getRegionInvocationBounds(operands, bounds);
    std::size_t regionNumber = frame.region->getRegionNumber();
    if (regionNumber >= bounds.size()) {
      return std::nullopt;
    }
    const InvocationBounds &regionBounds = bounds[regionNumber];
    auto maybeUpperBound = regionBounds.getUpperBound();
    SmallVector<RegionSuccessor> entrySuccessors;
    branch.getEntrySuccessorRegions(operands, entrySuccessors);
    if (entrySuccessors.size() == 1) {
      Region *selectedRegion = entrySuccessors.front().getSuccessor();
      if (!selectedRegion) {
        return 0;
      }
      if (selectedRegion != frame.region) {
        if (maybeUpperBound && *maybeUpperBound == 0) {
          return 0;
        }
        if (!isRegionReachable(branch, *selectedRegion, *frame.region)) {
          return 0;
        }
        return std::nullopt;
      }
      if (maybeUpperBound && *maybeUpperBound == 1) {
        return 1;
      }

      SmallVector<RegionSuccessor> regionSuccessors;
      branch.getSuccessorRegions(*selectedRegion, regionSuccessors);
      if (llvm::all_of(regionSuccessors, [](RegionSuccessor successor) {
            return successor.isOperation();
          })) {
        return 1;
      }
      return std::nullopt;
    }

    if (!maybeUpperBound || regionBounds.getLowerBound() != *maybeUpperBound) {
      return std::nullopt;
    }
    return *maybeUpperBound;
  }

  std::optional<llvm::APInt> getSCFForTripCount(
      scf::ForOp forOp,
      const llvm::DenseMap<Value, llvm::APInt> &inductionValues) const {
    if (std::optional<llvm::APInt> maybeTripCount =
            forOp.getStaticTripCount()) {
      return maybeTripCount;
    }

    IntegerExpressionEvaluator integerEvaluator =
        createIntegerEvaluator(inductionValues);
    std::optional<llvm::APInt> maybeLowerBound =
        integerEvaluator.evaluate(forOp.getLowerBound());
    std::optional<llvm::APInt> maybeUpperBound =
        integerEvaluator.evaluate(forOp.getUpperBound());
    std::optional<llvm::APInt> maybeStep =
        integerEvaluator.evaluate(forOp.getStep());
    if (!maybeLowerBound || !maybeUpperBound || !maybeStep ||
        maybeStep->isZero()) {
      return std::nullopt;
    }

    IntegerAttr lowerBoundAttr =
        IntegerAttr::get(forOp.getLowerBound().getType(), *maybeLowerBound);
    IntegerAttr upperBoundAttr =
        IntegerAttr::get(forOp.getUpperBound().getType(), *maybeUpperBound);
    IntegerAttr stepAttr =
        IntegerAttr::get(forOp.getStep().getType(), *maybeStep);
    return constantTripCount(lowerBoundAttr, upperBoundAttr, stepAttr,
                             /*isSigned=*/!forOp.getUnsignedCmp(),
                             scf::computeUbMinusLb);
  }

  // Evaluate a proven loop's trip count for the current induction environment; bounds are re-evaluated on every call.
  std::optional<std::uint64_t> evaluateNaturalLoopTripCount(
      const NaturalLoopInfo &info,
      const llvm::DenseMap<Value, llvm::APInt> &inductionValues) const {
    IntegerExpressionEvaluator integerEvaluator =
        createIntegerEvaluator(inductionValues);
    std::optional<llvm::APInt> maybeLowerBound =
        integerEvaluator.evaluate(info.lowerBoundOperand);
    std::optional<llvm::APInt> maybeBound =
        integerEvaluator.evaluate(info.boundOperand);
    if (!maybeLowerBound || !maybeBound) {
      return std::nullopt;
    }

    std::optional<llvm::APInt> maybeStep;
    for (Value stepOperand : info.stepOperands) {
      std::optional<llvm::APInt> maybeThisStep =
          integerEvaluator.evaluate(stepOperand);
      if (!maybeThisStep) {
        return std::nullopt;
      }
      if (maybeStep && *maybeStep != *maybeThisStep) {
        return std::nullopt;
      }
      maybeStep = maybeThisStep;
    }
    if (!maybeStep || maybeStep->isZero()) {
      return std::nullopt;
    }
    bool signMatchesDirection =
        info.ascending ? (!info.isSigned || maybeStep->isStrictlyPositive())
                       : maybeStep->isNegative();
    if (!signMatchesDirection) {
      return std::nullopt;
    }

    Type type = info.lowerBoundOperand.getType();
    IntegerAttr lowerBoundAttr = IntegerAttr::get(type, *maybeLowerBound);
    IntegerAttr upperBoundAttr = IntegerAttr::get(type, *maybeBound);
    IntegerAttr stepAttr = IntegerAttr::get(type, *maybeStep);
    std::optional<llvm::APInt> maybeTripCount = constantTripCount(
        lowerBoundAttr, upperBoundAttr, stepAttr, info.isSigned,
        scf::computeUbMinusLb);
    if (!maybeTripCount || maybeTripCount->getActiveBits() > 64) {
      return std::nullopt;
    }
    // Every provable shape is a do-while: it runs at least once even if the bound-based trip count is zero.
    std::uint64_t standardTripCount = maybeTripCount->getZExtValue();
    return std::max<std::uint64_t>(1, standardTripCount);
  }

  std::optional<std::uint64_t> getLoopTripCount(
      LoopLikeOpInterface loop,
      const llvm::DenseMap<Value, llvm::APInt> &inductionValues) const {
    std::optional<llvm::APInt> maybeTripCount = loop.getStaticTripCount();
    if (!maybeTripCount) {
      if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation())) {
        maybeTripCount = getSCFForTripCount(forOp, inductionValues);
      }
    }
    if (!maybeTripCount || maybeTripCount->getActiveBits() > 64) {
      return std::nullopt;
    }
    return maybeTripCount->getZExtValue();
  }

  std::optional<std::uint64_t>
  countInsideFrame(ArrayRef<ControlFrame> frames, std::size_t frameIndex,
                   llvm::DenseMap<Value, llvm::APInt> &inductionValues,
                   EnumerationBudget &enumerationBudget) {
    ControlFrame frame = frames[frameIndex];
    std::optional<std::uint64_t> maybeBlockCount = getExactBlockInvocationCount(
        *frame.region, frame.targetBlock, inductionValues);
    if (!maybeBlockCount || *maybeBlockCount == 0) {
      return maybeBlockCount;
    }
    std::optional<std::uint64_t> maybeNestedCount = countExecutions(
        frames, frameIndex + 1, inductionValues, enumerationBudget);
    return maybeNestedCount
               ? llvm::checkedMulUnsigned(*maybeBlockCount, *maybeNestedCount)
               : std::nullopt;
  }

  std::optional<std::uint64_t>
  countExecutions(ArrayRef<ControlFrame> frames, std::size_t frameIndex,
                  llvm::DenseMap<Value, llvm::APInt> &inductionValues,
                  EnumerationBudget &enumerationBudget) {
    if (frameIndex == frames.size()) {
      return 1;
    }

    ControlFrame frame = frames[frameIndex];
    auto loop = dyn_cast<LoopLikeOpInterface>(frame.parent);
    SmallVector<Region *> loopRegions;
    if (loop) {
      loopRegions = loop.getLoopRegions();
    }
    if (!loop || !llvm::is_contained(loopRegions, frame.region)) {
      std::optional<std::uint64_t> maybeInvocationCount =
          getExactRegionInvocationCount(frame, inductionValues);
      if (!maybeInvocationCount) {
        return std::nullopt;
      }
      if (*maybeInvocationCount == 0) {
        return 0;
      }
      std::optional<std::uint64_t> maybeNestedCount = countInsideFrame(
          frames, frameIndex, inductionValues, enumerationBudget);
      return maybeNestedCount ? llvm::checkedMulUnsigned(*maybeInvocationCount,
                                                         *maybeNestedCount)
                              : std::nullopt;
    }

    std::optional<std::uint64_t> maybeTripCount =
        getLoopTripCount(loop, inductionValues);
    if (!maybeTripCount) {
      return std::nullopt;
    }

    // A trip count does not define the invocation count of each region in a
    // multi-region loop.
    if (loopRegions.size() != 1) {
      return std::nullopt;
    }
    if (*maybeTripCount == 0) {
      return 0;
    }

    // Try multiplication before enumeration. Passing the same budget charges
    // nested enumeration even when this attempt cannot prove a count.
    std::optional<std::uint64_t> maybeNestedCount = countInsideFrame(
        frames, frameIndex, inductionValues, enumerationBudget);
    if (maybeNestedCount) {
      return llvm::checkedMulUnsigned(*maybeTripCount, *maybeNestedCount);
    }

    auto forOp = dyn_cast<scf::ForOp>(frame.parent);
    if (!forOp || !enumerationBudget.canConsume(*maybeTripCount)) {
      return std::nullopt;
    }
    IntegerExpressionEvaluator integerEvaluator =
        createIntegerEvaluator(inductionValues);
    std::optional<llvm::APInt> maybeInductionValue =
        integerEvaluator.evaluate(forOp.getLowerBound());
    std::optional<llvm::APInt> maybeStep =
        integerEvaluator.evaluate(forOp.getStep());
    if (!maybeInductionValue || !maybeStep) {
      return std::nullopt;
    }

    assert(maybeInductionValue->getBitWidth() == maybeStep->getBitWidth() &&
           "loop bounds and step must have the same bit width");
    llvm::scope_exit restoreInductionValue(
        [&] { inductionValues.erase(forOp.getInductionVar()); });
    std::uint64_t total = 0;
    for (std::uint64_t iteration = 0; iteration < *maybeTripCount;
         ++iteration) {
      if (!enumerationBudget.tryConsume()) {
        return std::nullopt;
      }
      inductionValues[forOp.getInductionVar()] = *maybeInductionValue;
      maybeNestedCount = countInsideFrame(frames, frameIndex, inductionValues,
                                          enumerationBudget);
      if (!maybeNestedCount) {
        return std::nullopt;
      }
      std::optional<std::uint64_t> maybeNextTotal =
          llvm::checkedAddUnsigned(total, *maybeNestedCount);
      if (!maybeNextTotal) {
        return std::nullopt;
      }
      total = *maybeNextTotal;
      *maybeInductionValue += *maybeStep;
    }
    return total;
  }

  Region &rootRegion;
  SymbolValueEvaluator symbolValueEvaluator;
  RegionInvocationCountEvaluator regionInvocationCountEvaluator;
  Options options;
  std::unique_ptr<DataFlowSolver> dataFlowSolver;
  llvm::DenseMap<Block *, std::optional<std::uint64_t>>
      blockExecutionCountCache;
  /// Reuse block counts when different induction values select the same edges.
  llvm::DenseMap<Region *, llvm::DenseMap<BlockFlowKey, BlockCountResult>>
      blockCountCache;
};

ExecutionCountAnalysis::ExecutionCountAnalysis(
    Region &rootRegion, SymbolValueEvaluator symbolValueEvaluator,
    RegionInvocationCountEvaluator regionInvocationCountEvaluator)
    : ExecutionCountAnalysis(rootRegion, std::move(symbolValueEvaluator),
                             std::move(regionInvocationCountEvaluator),
                             Options{}) {}

ExecutionCountAnalysis::ExecutionCountAnalysis(
    Region &rootRegion, SymbolValueEvaluator symbolValueEvaluator,
    RegionInvocationCountEvaluator regionInvocationCountEvaluator,
    Options options)
    : impl(std::make_unique<Impl>(rootRegion, std::move(symbolValueEvaluator),
                                  std::move(regionInvocationCountEvaluator),
                                  options)) {}

ExecutionCountAnalysis::~ExecutionCountAnalysis() = default;

ExecutionCountAnalysis::ExecutionCountAnalysis(
    ExecutionCountAnalysis &&) noexcept = default;

ExecutionCountAnalysis &
ExecutionCountAnalysis::operator=(ExecutionCountAnalysis &&) noexcept = default;

std::optional<std::uint64_t>
ExecutionCountAnalysis::getExecutionCount(Operation *operation) {
  return impl->getExecutionCount(operation);
}

} // namespace mlir::tt
