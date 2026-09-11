# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Transfer graph used by the activation all-gather."""

from itertools import product

import ttl


def make_ring_graph(device_domain, mesh_shape):
    coordinates = tuple(product(*(range(extent) for extent in mesh_shape)))
    return ttl.TransferGraph.edges(
        device_domain,
        edges=tuple(
            (
                coordinates[source_index],
                coordinates[(source_index + 1) % len(coordinates)],
            )
            for source_index in range(len(coordinates))
        ),
    )
