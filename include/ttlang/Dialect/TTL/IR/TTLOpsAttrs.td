// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPSATTRS_TD
#define TTLANG_DIALECT_TTL_IR_TTLOPSATTRS_TD

include "ttlang/Dialect/TTL/IR/TTLBase.td"
include "ttlang/Dialect/TTL/IR/TTLOpsEnums.td"

//===----------------------------------------------------------------------===//
// TTL attribute definitions
//===----------------------------------------------------------------------===//

def TTL_SliceAttr : TTL_Attr<"Slice", "slice"> {
  let summary = "Slice specification for core range";
  let parameters = (ins
    "int64_t":$start,
    "int64_t":$stop,
    "int64_t":$step
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
  let description = [{
    Encodes a Python slice(start, stop, step) for pipe multicast ranges.
    Half-open interval: [start, stop).
  }];
}

def TTL_LayoutAttr : TTL_Attr<"Layout", "layout"> {
  let summary = "Tensor layout encoding for grid distribution and memory placement";
  let description = [{
    Encodes how a tensor is distributed across the device grid and placed in
    memory. Used as the encoding attribute on RankedTensorType.

    This replaces the dependency on TTNNLayoutAttr from the TTNN dialect,
    providing the minimal layout information needed by the tt-lang compiler:
    - Logical tensor shape
    - Tile element type
    - Buffer type (L1, DRAM, etc.)
    - Grid shape for distribution
    - Memory layout (interleaved, sharded, etc.)
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "BufferType":$bufferType,
    ArrayRefParameter<"int64_t">:$grid,
    "TensorMemoryLayout":$memoryLayout
  );

  let assemblyFormat = "`<` `shape` `=` `[` $shape `]` `,`"
                       " `element_type` `=` $elementType `,`"
                       " `buffer` `=` $bufferType `,`"
                       " `grid` `=` `[` $grid `]` `,`"
                       " `memory` `=` $memoryLayout `>`";

  let genVerifyDecl = 1;
}

#endif // TTLANG_DIALECT_TTL_IR_TTLOPSATTRS_TD
