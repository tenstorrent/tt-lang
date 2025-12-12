// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPSENUMS_H
#define TTLANG_DIALECT_TTL_IR_TTLOPSENUMS_H

#include <cstdint>
#include <optional>
#include <string>

#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

// Wrapper for TableGen-generated TTL enum declarations, will be used for future
// enum definitions.
#include "ttlang/Dialect/TTL/IR/TTLOpsEnums.h.inc"

#endif // TTLANG_DIALECT_TTL_IR_TTLOPSENUMS_H
