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

import torch
import ttnn

device = ttnn.open_device(device_id=0)

try:
    # spec:begin
    def from_torch(tensor: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            tensor,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def shape_in_tiles(tensor: ttnn.Tensor) -> list[int]:
        padded_shape = list(tensor.padded_shape)
        tile_shape = list(tensor.tile.tile_shape)
        return padded_shape[:-2] + [
            dim // tile_dim for dim, tile_dim in zip(padded_shape[-2:], tile_shape)
        ]

    assert shape_in_tiles(from_torch(torch.randn(()))) == [1, 1]
    assert shape_in_tiles(from_torch(torch.randn((128)))) == [1, 4]
    assert shape_in_tiles(from_torch(torch.randn((1, 128)))) == [1, 4]
    assert shape_in_tiles(from_torch(torch.randn((32, 128)))) == [1, 4]
    assert shape_in_tiles(from_torch(torch.randn((128, 1)))) == [4, 1]
    assert shape_in_tiles(from_torch(torch.randn((128, 32)))) == [4, 1]
    assert shape_in_tiles(from_torch(torch.randn((2, 128, 32)))) == [2, 4, 1]
    assert shape_in_tiles(from_torch(torch.randn((2, 2, 128, 32)))) == [2, 2, 4, 1]
    assert shape_in_tiles(from_torch(torch.randn((2, 2, 120, 30)))) == [2, 2, 4, 1]
    # spec:end

finally:
    ttnn.close_device(device)
