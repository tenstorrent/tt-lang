// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang-c/TTLAttrs.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

using namespace mlir;
using namespace mlir::tt::ttl;

//===----------------------------------------------------------------------===//
// TTL SliceAttr
//===----------------------------------------------------------------------===//

bool ttlangMlirAttributeIsATTLSliceAttr(MlirAttribute attr) {
  return isa<SliceAttr>(unwrap(attr));
}

MlirAttribute ttlangTTLSliceAttrGet(MlirContext ctx, int64_t start,
                                     int64_t stop, int64_t step) {
  return wrap(SliceAttr::get(unwrap(ctx), start, stop, step));
}

int64_t ttlangTTLSliceAttrGetStart(MlirAttribute attr) {
  return cast<SliceAttr>(unwrap(attr)).getStart();
}

int64_t ttlangTTLSliceAttrGetStop(MlirAttribute attr) {
  return cast<SliceAttr>(unwrap(attr)).getStop();
}

int64_t ttlangTTLSliceAttrGetStep(MlirAttribute attr) {
  return cast<SliceAttr>(unwrap(attr)).getStep();
}
