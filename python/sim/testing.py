# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Testing and validation utilities for simulation.
"""

from .ttnnsim import Tensor, isclose


def assert_pcc(
    tensor_a: Tensor,
    tensor_b: Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    # tensors should be equal
    assert tensor_a.shape == tensor_b.shape, "Tensors must have the same shape"
    assert tensor_a.dtype == tensor_b.dtype, "Tensors must have the same dtype"
    close_result = isclose(tensor_a, tensor_b, rtol=rtol, atol=atol)
    assert close_result.to_torch().all().item(), "Tensor values are not close enough"
