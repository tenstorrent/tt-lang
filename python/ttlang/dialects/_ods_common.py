# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Reuse upstream MLIR utilities; keep a namespaced shim for generated bindings.
from ttmlir.dialects import _ods_common as _upstream

# Re-export the upstream cext and all public helpers so generated stubs work.
_cext = _upstream._cext
__all__ = ["_cext"]
__all__.extend(_upstream.__all__)
globals().update({k: getattr(_upstream, k) for k in _upstream.__all__})
