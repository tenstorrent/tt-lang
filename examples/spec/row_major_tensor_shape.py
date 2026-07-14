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
def from_torch(tensor: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )


def row_major_shape(tensor: ttnn.Tensor) -> list[int]:
    return list(tensor.padded_shape)


row_major_shape(from_torch(torch.randn(())))  #              prints [1]
row_major_shape(from_torch(torch.randn((128))))  #           prints [128]
row_major_shape(from_torch(torch.randn((1, 128))))  #        prints [1, 128]
row_major_shape(from_torch(torch.randn((32, 128))))  #       prints [32, 128]
row_major_shape(from_torch(torch.randn((128, 1))))  #        prints [128, 1]
row_major_shape(from_torch(torch.randn((128, 32))))  #       prints [128, 32]
row_major_shape(from_torch(torch.randn((2, 128, 32))))  #    prints [2, 128, 32]
row_major_shape(from_torch(torch.randn((2, 2, 128, 32))))  # prints [2, 2, 128, 32]
row_major_shape(from_torch(torch.randn((2, 2, 120, 30))))  # prints [2, 2, 120, 30]
# spec:end
