# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Example source for docs/sphinx/specs/TTLangSpecification.md.
#
# The lines between the "spec:begin" and "spec:end" markers below are included
# verbatim in the specification. Regenerate the specification after editing:
#
#     python docs/sphinx/specs/build_spec.py
#
# Everything outside the markers (imports, scaffolding) exists so the file can
# stand on its own and is not copied into the specification.

import math

import torch

import ttl
import ttnn

# spec:begin
# for (8, 8) single chip or SPMD grid gets x = [0, 64)
x = ttl.node(dims=1)

# for (8, 8, 8) multi-chip grid gets x = [0, 8), y = [0, 64)
x, y = ttl.node(dims=2)

# for (8, 8) single-chip or SPMD grid gets x = [0, 8), y = [0, 8), z = 0
x, y, z = ttl.node(dims=3)
# spec:end
