// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Pipe Transfer Expansion
//===----------------------------------------------------------------------===//
//
// This file declares conversion from high-level pipe copies to explicit pipe
// transfer operations.
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPETRANSFEREXPANSION_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPETRANSFEREXPANSION_H

#include "PipeGraph.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt {
class ValueOriginAnalysis;
}

namespace mlir::tt::ttl {

/// Return the transfer contract shared by every record of a verified record
/// set.
FailureOr<PipeTransferContract>
getPipeTransferContractForRecords(PipeNetRecordsAttr records);

/// Return the contract shared by every possible value of a pipe operand.
///
/// Create and selected-pipe operations preserve an explicit collective
/// contract. A block argument has no defining pipe op, so its type supplies
/// the contract.
FailureOr<PipeTransferContract>
getPipeTransferContractForPipeValue(ValueOriginAnalysis &analysis, Value pipe);

/// Replace high-level pipe copies and waits with explicit pipe transfer IR.
LogicalResult expandPipeTransfers(ModuleOp module,
                                  ValueOriginAnalysis &analysis);

/// Replace high-level copies and waits whose pipe operand has `ttl.pipe` type.
/// Record-selected callback operands remain high-level IR for later expansion.
LogicalResult expandStaticPipeTransfers(ModuleOp module,
                                        ValueOriginAnalysis &analysis);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPETRANSFEREXPANSION_H
