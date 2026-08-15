//===- CommonRuntimeArgLayout.h - Common argument indices -----*- C++ -*-===//
//
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file declares the common runtime argument layout that TTL lowering and
// the host runtime implement.
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_COMMONRUNTIMEARGLAYOUT_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_COMMONRUNTIMEARGLAYOUT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <cstdint>

namespace mlir::tt::ttl {

/// Computes common runtime argument indices for one kernel function.
///
/// Tensor buffer addresses come first, followed by computed receiver DFB
/// bases, compiler-managed PipeNet resources, an optional fabric unique-runtime
/// argument base, and logical device coordinates. These segments form the
/// compiler-defined prefix. Per-kernel extra arguments follow the prefix.
class CommonRuntimeArgLayout {
public:
  /// Construct the layout from metadata attached to the function and module.
  explicit CommonRuntimeArgLayout(func::FuncOp function);

  /// Construct the layout with a planned computed receiver DFB base count.
  CommonRuntimeArgLayout(func::FuncOp function,
                         int64_t computedReceiverDFBBaseCount);

  /// Return the argument index for one computed receiver DFB base.
  int64_t getComputedReceiverDFBBaseIndex(int64_t ordinal) const;

  /// Return the argument index for one compiler-managed PipeNet resource.
  int64_t getPipeResourceIndex(int64_t ordinal) const;

  /// Return the common argument index containing the fabric argument base.
  int64_t getFabricRuntimeArgBaseIndex() const;

  /// Return the argument index for one logical device coordinate.
  int64_t getDeviceCoordinateIndex(int64_t ordinal) const;

private:
  int64_t computedReceiverDFBBaseArgIndex = 0;
  int64_t computedReceiverDFBBaseCount = 0;
  int64_t pipeResourceBaseArgIndex = 0;
  int64_t pipeResourceCount = 0;
  int64_t fabricRuntimeArgBaseIndex = 0;
  bool hasFabricRuntimeArgBase = false;
  int64_t deviceCoordinateBaseArgIndex = 0;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_COMMONRUNTIMEARGLAYOUT_H
