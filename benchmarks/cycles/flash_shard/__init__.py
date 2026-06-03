# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-core cycle A/B for the flash-decode shard.

``ttl.py`` measures the ttl.ops.flash_mla shard; ``metal.py`` measures the
metal ``compute_sdpa_chunk`` primitive (kernels under ``kernels/``) on the same
per-core problem (``shapes.py``). ``__main__`` runs both and prints the ratio.
A third, hand-optimized tt-lang variant -- the codegen we think ttl could reach
-- is a planned addition here.
"""
