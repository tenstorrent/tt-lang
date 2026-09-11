# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Invalid device indexing and node endpoint selections."""

import pytest

from ttl.domains import DeviceDomain


@pytest.mark.parametrize("index", [True, 1.5, None, "row"])
def test_index_rejects_non_integer_non_slice(index):
    with pytest.raises(TypeError):
        DeviceDomain((8, 4))[index]


@pytest.mark.parametrize("index", [(8, 0), (0, 4), (0, 0, 0)])
def test_index_rejects_out_of_bounds_or_extra_axes(index):
    with pytest.raises(IndexError):
        DeviceDomain((8, 4))[index]


@pytest.mark.parametrize("node", [(-1, 0), (0, -1), (True, 0), (0.5, 0)])
def test_node_coordinates_are_validated(node):
    with pytest.raises((TypeError, ValueError)):
        DeviceDomain((8, 4))[0, :].at_node(*node)


def test_selection_rejects_incompatible_parent_domains():
    first = DeviceDomain((8, 4))
    second = DeviceDomain((2, 8, 4))
    with pytest.raises(ValueError, match="parent domain"):
        first.select([second[0, 0, 0]])
    with pytest.raises(ValueError, match="parent domain"):
        first[:] | second[:]
