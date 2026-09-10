// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPERECEIVEBATCHING_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPERECEIVEBATCHING_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir::tt::ttl {

class PipeGraph;
struct PipeForeachLoweringInfo;
struct PipeResourcePlan;

/// Mark record loops whose complete receive sequence fits initially empty DFB
/// storage, using the graph's producer ownership and computed-address proofs.
/// This runs before conversion erases logical DFB and transfer identities.
void annotateInitialPipeReceiveBatches(
    ModuleOp module, const PipeForeachLoweringInfo &foreachInfo,
    const PipeGraph &graph, const PipeResourcePlan &resources);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPERECEIVEBATCHING_H
