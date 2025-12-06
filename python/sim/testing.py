# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Testing and validation utilities for simulation.
"""

import torch
from . import torch_utils as tu


def assert_pcc(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> None:
    # tensors should be equal
    assert tensor_a.shape == tensor_b.shape, "Tensors must have the same shape"
    assert tensor_a.dtype == tensor_b.dtype, "Tensors must have the same dtype"
    assert tensor_a.device == tensor_b.device, "Tensors must be on the same device"
    assert tu.allclose(tensor_a, tensor_b), "Tensor values are not close enough"
