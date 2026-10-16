// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBNetworkPlanning.h"

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/DFBLogicalIdentityAnalysis.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstdint>
#include <string>

namespace mlir::tt::ttl {

namespace {

/// Kernel-side facts about one DFB that network records name.
struct DFBEndpoints {
  /// `ttl.bind_cb` declarations of the DFB, in module order.
  SmallVector<BindCBOp> bindings;

  /// First record (in module order) that names this DFB.
  Operation *firstRecord = nullptr;

  /// Kernel ops that reserve or push the DFB, in module order. One producer
  /// kernel usually contributes several.
  llvm::SetVector<Operation *> producerOps;

  /// Kernel ops that wait on or pop the DFB, in module order.
  llvm::SetVector<Operation *> consumerOps;

  /// Native calls in kernels that take the DFB, in module order.
  llvm::SetVector<OpaqueCallOp> nativeCalls;
};

/// One DFB that a record names, with the end the record holds.
struct DFBRecordUse {
  int64_t id;
  DFBNetworkRole role;
};

} // namespace

/// Appends `handle` to `uses` when it names a DFB rather than a merged handle.
static void appendDFBUse(Attribute handle, DFBNetworkRole role,
                         SmallVectorImpl<DFBRecordUse> &uses) {
  if (auto id = dyn_cast<IntegerAttr>(handle)) {
    uses.push_back({id.getInt(), role});
  }
}

/// Returns the DFBs that `record` names, in operand order.
static SmallVector<DFBRecordUse> getRecordDFBUses(Operation *record) {
  SmallVector<DFBRecordUse> uses;
  TypeSwitch<Operation *>(record)
      .Case<DFBForkOp, DFBSplitOp>([&](auto distributor) {
        appendDFBUse(distributor.getSource(), DFBNetworkRole::Source, uses);
        for (int64_t output : distributor.getOutputs()) {
          uses.push_back({output, DFBNetworkRole::Output});
        }
      })
      .Case<DFBMergeOp>([&](DFBMergeOp merge) {
        for (Attribute input : merge.getInputs()) {
          appendDFBUse(input, DFBNetworkRole::Input, uses);
        }
      });
  return uses;
}

StringRef stringifyDFBNetworkRole(DFBNetworkRole role) {
  switch (role) {
  case DFBNetworkRole::Source:
    return "source";
  case DFBNetworkRole::Input:
    return "input";
  case DFBNetworkRole::Output:
    return "output";
  }
  llvm_unreachable("unknown DFB network role");
}

/// Returns how a diagnostic names `role`, e.g. "the source of".
static StringRef describeRole(DFBNetworkRole role) {
  switch (role) {
  case DFBNetworkRole::Source:
    return "the source of";
  case DFBNetworkRole::Input:
    return "an input of";
  case DFBNetworkRole::Output:
    return "an output of";
  }
  llvm_unreachable("unknown DFB network role");
}

DFBNetworkPlanningResult DFBNetworkPlanner::build() const {
  ModuleOp module = this->module;
  SmallVector<DFBNetworkOp> networks;
  module.walk([&](DFBNetworkOp network) { networks.push_back(network); });
  // Without networks the module has nothing to expand, and its DFB
  // declarations are left to the passes that own them.
  if (networks.empty()) {
    return DFBNetworkPlanningResult::planned(DFBNetworkPlan());
  }

  DFBLogicalIdentityAnalysis identities(module);
  if (!identities.succeeded()) {
    return DFBNetworkPlanningResult::invalidIR(
        identities.getErrorOperation(), identities.getErrorMessage().str());
  }

  llvm::MapVector<int64_t, DFBEndpoints> endpointsById;
  for (const DFBLogicalIdentityAssignment &assignment :
       identities.getAssignments()) {
    endpointsById[assignment.logicalId].bindings.push_back(
        assignment.declaration);
  }

  module.walk([&](DFBAccessOpInterface access) {
    if (!getEnclosingKernelThread(access)) {
      return;
    }
    for (const DFBProtocolEffect &effect : access.getDFBProtocolEffects()) {
      FailureOr<int64_t> id = identities.getLogicalId(effect.dfb);
      if (failed(id)) {
        continue;
      }
      DFBEndpoints &endpoints = endpointsById[*id];
      // A native call can declare several effects on one DFB, such as a wait
      // and a pop; the set lists it once.
      (isProducerDFBProtocolEffect(effect.kind) ? endpoints.producerOps
                                                : endpoints.consumerOps)
          .insert(access);
    }
  });

  // A native call may access a DFB without declaring its effects, so calls
  // are collected from their operands rather than their effects.
  module.walk([&](OpaqueCallOp call) {
    if (!getEnclosingKernelThread(call)) {
      return;
    }
    for (Value dfb : call.getDFBDependencyOperands()) {
      FailureOr<int64_t> id = identities.getLogicalId(dfb);
      if (failed(id)) {
        continue;
      }
      endpointsById[*id].nativeCalls.insert(call);
    }
  });

  for (DFBNetworkOp network : networks) {
    for (Operation &record : network.getBody()->getOperations()) {
      for (auto [id, role] : getRecordDFBUses(&record)) {
        if (!endpointsById.contains(id)) {
          return DFBNetworkPlanningResult::invalidIR(
              &record,
              ("DFB " + Twine(id) + " has no ttl.bind_cb in any kernel").str());
        }

        DFBEndpoints &endpoints = endpointsById[id];
        if (endpoints.firstRecord == nullptr) {
          endpoints.firstRecord = &record;
        }

        auto firstNetwork =
            cast<DFBNetworkOp>(endpoints.firstRecord->getParentOp());
        if (firstNetwork != network) {
          return DFBNetworkPlanningResult::invalidIR(
              &record,
              ("DFB " + Twine(id) + " is used by more than one network").str(),
              endpoints.firstRecord,
              ("first used in network @" + firstNetwork.getSymName() + " here")
                  .str());
        }

        // The record holds the consumer end of a source or input and the
        // producer end of an output; a kernel holding the same end would
        // make two producers or two consumers.
        bool recordConsumes = role != DFBNetworkRole::Output;
        const llvm::SetVector<Operation *> &kernelOps =
            recordConsumes ? endpoints.consumerOps : endpoints.producerOps;
        if (!kernelOps.empty()) {
          return DFBNetworkPlanningResult::invalidIR(
              &record,
              ("DFB " + Twine(id) + " is " + describeRole(role) + " @" +
               cast<SymbolOpInterface>(record).getName() + ", but a kernel " +
               (recordConsumes ? "consumes" : "produces") + " it")
                  .str(),
              kernelOps.front(),
              recordConsumes ? "consumed here" : "produced here");
        }
      }
    }
  }

  // Every network is valid. Report each feature this compiler cannot expand
  // yet, so one compilation lists all of them.
  SmallVector<DFBNetworkRejection> rejections;
  auto reject = [&](Operation *operation, const Twine &reason) {
    rejections.push_back({operation, reason.str()});
  };
  for (DFBNetworkOp network : networks) {
    Block &body = *network.getBody();
    if (body.getOperations().size() > 1) {
      reject(network, "@" + network.getSymName() +
                          ": networks with more than one record are not "
                          "supported yet; connect the operators through a "
                          "kernel");
    }
    for (Operation &record : body) {
      if (isa<DFBMergeOp>(record)) {
        reject(&record, "merge records are not supported yet");
      }
      if (auto fork = dyn_cast<DFBForkOp>(record)) {
        DFBNetworkForkStorage storage = fork.getStorage().getValue();
        if (storage != DFBNetworkForkStorage::Replicated) {
          reject(fork, stringifyDFBNetworkForkStorage(storage) +
                           " fork storage is not supported yet; use storage "
                           "= replicated");
        }
      }
      for (auto [id, role] : getRecordDFBUses(&record)) {
        const DFBEndpoints &endpoints = endpointsById.find(id)->second;
        auto grouped = llvm::find_if(endpoints.bindings, [](BindCBOp binding) {
          return binding.getAllocationGroup().has_value();
        });
        if (grouped != endpoints.bindings.end()) {
          reject(*grouped, "DFB " + Twine(id) + " of network @" +
                               network.getSymName() +
                               " has an allocation group, which is not "
                               "supported yet");
        }
        if (!endpoints.nativeCalls.empty()) {
          reject(endpoints.nativeCalls.front(),
                 "native calls on DFB " + Twine(id) + " of network @" +
                     network.getSymName() + " are not supported yet");
        }
      }
    }
  }
  if (!rejections.empty()) {
    return DFBNetworkPlanningResult::rejected(std::move(rejections));
  }

  DFBNetworkPlan plan;
  for (DFBNetworkOp network : networks) {
    DFBNetworkExpansion &expansion = plan.networks.emplace_back();
    expansion.network = network;
    for (Operation &record : *network.getBody()) {
      DFBNetworkRecordPlan &recordPlan = expansion.records.emplace_back();
      recordPlan.record = &record;
      for (auto [id, role] : getRecordDFBUses(&record)) {
        const DFBEndpoints &endpoints = endpointsById.find(id)->second;
        const llvm::SetVector<Operation *> &kernelOps =
            role == DFBNetworkRole::Output ? endpoints.consumerOps
                                           : endpoints.producerOps;
        recordPlan.endpoints.push_back(
            {id, role, endpoints.bindings,
             SmallVector<Operation *>(kernelOps.begin(), kernelOps.end())});
      }
    }
  }
  return DFBNetworkPlanningResult::planned(std::move(plan));
}

} // namespace mlir::tt::ttl
