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
# ttl.node() resolves against the current node, so the marked lines run per
# node during operation setup.

import ttl


@ttl.operation(grid=(8, 8))
def node_example() -> None:
    # spec:begin
    # for (8, 8) single chip or SPMD grid gets x = [0, 64)
    x = ttl.node(dims=1)

    # for (8, 8, 8) multi-chip grid gets x = [0, 8), y = [0, 64)
    x, y = ttl.node(dims=2)

    # for (8, 8) single-chip or SPMD grid gets x = [0, 8), y = [0, 8), z = 0
    x, y, z = ttl.node(dims=3)
    # spec:end

    # On a (8, 8) grid each form yields the current node's coordinates at that
    # rank: a linear index in [0, 64), then a pair in [0, 8) x [0, 8), then the
    # same pair with z == 0. The marked lines rebind x and y, so the first two
    # forms are read again here rather than left as a claim about values nothing
    # checks; the linear index and the pair have to agree about the node.
    assert 0 <= ttl.node(dims=1) < 64
    assert ttl.node(dims=1) == x * 8 + y
    assert ttl.node(dims=2) == (x, y)
    assert 0 <= x < 8 and 0 <= y < 8 and z == 0


node_example()
