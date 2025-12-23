# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Re-export the upstream MLIR Python extension from tt-mlir so generated
# bindings that expect `ttlang._mlir_libs._mlir` can reuse it without us
# shipping a duplicate binary.
import sys

from ttmlir._mlir_libs import (  # type: ignore
    _mlir,
    get_dialect_registry,
)

sys.modules[__name__ + "._mlir"] = _mlir
