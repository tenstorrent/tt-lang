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

# Eagerly register tt-lang dialects/passes into the shared global registry.
#
# Note that ttmlir’s site-initialization only probes for _site_initialize_{i}
# modules under *ttmlir._mlir_libs*. Since tt-lang’s initializer lives under
# *ttlang._mlir_libs*, it will not be automatically discovered. To preserve the
# intended “site_initialize” extension mechanism without modifying tt-mlir, we
# explicitly invoke our initializer here at import time.
try:
    from . import _site_initialize_1 as _ttlang_site_initialize_1

    _ttlang_site_initialize_1.register_dialects(get_dialect_registry())
except Exception:
    # Registration is best-effort; failures here should not prevent importing
    # the package, but may require explicit per-context registration.
    pass

try:
    from . import _ttlang as _ttlang_ext

    if hasattr(_ttlang_ext, "register_all_passes"):
        _ttlang_ext.register_all_passes()
except Exception:
    # Pass registration is best-effort; failures will be surfaced when
    # constructing pipelines.
    pass
