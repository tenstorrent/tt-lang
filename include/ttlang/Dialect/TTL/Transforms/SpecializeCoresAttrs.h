// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Attribute names shared by the two per-core specialization passes.
//
// Phase A (`ttl-specialize-plan`, TTL level) writes these; Phase B
// (`ttl-specialize-cores`, TTKernel level) reads them. They are deliberately
// dialect-neutral so they survive lowering between the two phases.
//===----------------------------------------------------------------------===//

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_SPECIALIZECORESATTRS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_SPECIALIZECORESATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir::tt::ttl {

/// Per-function attribute giving the `[gridX, gridY]` launch extent.
inline constexpr llvm::StringLiteral kOperationGridAttrName =
    "ttl.operation_grid";

/// Per-function specialization plan written by Phase A. An `ArrayAttr` of
/// `DictionaryAttr`, one per coordinate group, each with:
///   - `coords`: `DenseI64ArrayAttr` of flattened `[x0, y0, x1, y1, ...]`.
///   - `taken`:  `DenseBoolArrayAttr` indexed by `ttl.specialize_branch` id.
inline constexpr llvm::StringLiteral kSpecializePlanAttrName =
    "ttl.specialize_plan";

/// Marker set by Phase A on every `scf.if` that branches on a core coordinate.
/// The `i64` value indexes the per-group `taken` array in the plan.
inline constexpr llvm::StringLiteral kSpecializeBranchAttrName =
    "ttl.specialize_branch";

/// Per-clone attribute written by Phase B recording the `[x, y]` coordinates
/// the clone serves, as an array of length-2 `[x, y]` arrays.
inline constexpr llvm::StringLiteral kCoreCoordAttrName = "ttl.core_coord";

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_SPECIALIZECORESATTRS_H
