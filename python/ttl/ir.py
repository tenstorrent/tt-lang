# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Thin shim to reuse the upstream ttmlir.ir module so dialect bindings can
# import `ttlang.ir` just like ttmlir.
from ttmlir.ir import *  # noqa: F401,F403
