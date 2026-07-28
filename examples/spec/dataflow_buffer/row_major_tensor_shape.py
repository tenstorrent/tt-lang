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
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

    def row_major_shape(tensor: ttnn.Tensor) -> list[int]:
        return list(tensor.padded_shape)

    assert row_major_shape(from_torch(torch.randn(()))) == [1]
    assert row_major_shape(from_torch(torch.randn((128)))) == [128]
    assert row_major_shape(from_torch(torch.randn((1, 128)))) == [1, 128]
    assert row_major_shape(from_torch(torch.randn((32, 128)))) == [32, 128]
    assert row_major_shape(from_torch(torch.randn((128, 1)))) == [128, 1]
    assert row_major_shape(from_torch(torch.randn((128, 32)))) == [128, 32]
    assert row_major_shape(from_torch(torch.randn((2, 128, 32)))) == [2, 128, 32]
    assert row_major_shape(from_torch(torch.randn((2, 2, 128, 32)))) == [2, 2, 128, 32]
    assert row_major_shape(from_torch(torch.randn((2, 2, 120, 30)))) == [2, 2, 120, 30]
    # spec:end

finally:
    ttnn.close_device(device)
