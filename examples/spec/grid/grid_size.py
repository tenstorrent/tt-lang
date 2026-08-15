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
# Everything outside the markers (imports, the @ttl.operation wrapper, and the
# correctness check) exists so the file can run standalone; it is not copied
# into the specification. The marked lines are nested inside @ttl.operation and
# dedented on render, so these mechanics add nothing to the rendered text.
# ttl.grid_size() resolves against the operation's grid, so the marked lines
# run per node during operation setup.

import ttl


@ttl.operation(grid=(8, 8))
def grid_size_example() -> None:
    # spec:begin
    # for (8, 8) single chip or SPMD grid gets x_size = 64
    x_size = ttl.grid_size(dims=1)

    # for (8, 8, 8) multi-chip grid gets x_size = 8, y_size = 64
    x_size, y_size = ttl.grid_size(dims=2)

    # for (8, 8) single-chip or SPMD grid gets x_size = 8, y_size = 8, z_size = 1
    x_size, y_size, z_size = ttl.grid_size(dims=3)
    # spec:end

    # Grid size is node-independent; on a (8, 8) grid the successive forms yield
    # 64, then (8, 8), then (8, 8, 1). Each is asserted: the marked lines rebind
    # x_size and y_size, so the first two forms are read again here rather than
    # left as a claim about values nothing checks.
    assert ttl.grid_size(dims=1) == 64
    assert ttl.grid_size(dims=2) == (8, 8)
    assert (x_size, y_size, z_size) == (8, 8, 1)


grid_size_example()
