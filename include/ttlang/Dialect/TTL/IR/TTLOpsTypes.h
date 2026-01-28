// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPSTYPES_H
#define TTLANG_DIALECT_TTL_IR_TTLOPSTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsEnums.h"

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <optional>

#define GET_TYPEDEF_CLASSES
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h.inc"

#endif // TTLANG_DIALECT_TTL_IR_TTLOPSTYPES_H
