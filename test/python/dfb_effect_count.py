# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: %python %s

"""Verify bounded arithmetic used before DFB repeat materialization."""

from ttl._src.ttl_ast import (
    _MAX_EXPANDED_EXTERNAL_DFB_EFFECTS,
    _saturating_add_expanded_dfb_effect_count,
)


LIMIT = _MAX_EXPANDED_EXTERNAL_DFB_EFFECTS
OVER_LIMIT = LIMIT + 1


# Exact-boundary counts remain exact; any additional action saturates.
assert _saturating_add_expanded_dfb_effect_count(0, 1, LIMIT) == LIMIT
assert _saturating_add_expanded_dfb_effect_count(1, 1, LIMIT) == OVER_LIMIT
assert _saturating_add_expanded_dfb_effect_count(LIMIT // 2, 1, LIMIT // 2) == LIMIT
assert (
    _saturating_add_expanded_dfb_effect_count(LIMIT // 2, 1, LIMIT // 2 + 1)
    == OVER_LIMIT
)

# Zero copies preserve the existing state, including an earlier rejection.
assert _saturating_add_expanded_dfb_effect_count(LIMIT, 1, 0) == LIMIT
assert _saturating_add_expanded_dfb_effect_count(OVER_LIMIT, 1, 0) == OVER_LIMIT

# Huge values must reject without forming their product.
HUGE_COUNT = 10**1000
assert (
    _saturating_add_expanded_dfb_effect_count(0, HUGE_COUNT, HUGE_COUNT) == OVER_LIMIT
)
