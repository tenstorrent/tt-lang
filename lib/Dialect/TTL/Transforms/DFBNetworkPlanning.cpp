// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBNetworkPlanning.h"

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

namespace mlir::tt::ttl {

DFBNetworkPlanningResult DFBNetworkPlanner::build() const {
  ModuleOp module = this->module;
  SmallVector<DFBNetworkRejection> rejections;
  module.walk([&](DFBNetworkOp network) {
    DFBNetworkRejection rejection;
    rejection.operation = network;
    rejection.reason =
        ("@" + network.getSymName() + ": DFB networks are not supported yet")
            .str();
    rejections.push_back(std::move(rejection));
  });

  if (!rejections.empty()) {
    return DFBNetworkPlanningResult::rejected(std::move(rejections));
  }
  return DFBNetworkPlanningResult::planned(DFBNetworkPlan());
}

} // namespace mlir::tt::ttl
