# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TensorAccessor for indexed tile-level access to tensor arguments in kernels."""

from typing import Optional, Any

from ttlang._src.tensor_registry import get_tensor_global_name


class TensorAccessor:
    """
    Provides indexed tile-level access to tensor arguments for DMA operations.

    TensorAccessor wraps a tensor function argument to enable indexed access
    for data movement operations between DRAM and L1. The tensor must be a
    top-level function argument (not a local variable).

    Under the hood, this creates a d2m.stream_layout operation with backing
    storage to enable efficient multi-buffered access patterns.

    Attributes:
        name: Global name of the wrapped tensor
        shape: Shape of the tensor
        dtype: Data type of the tensor (e.g., torch.float32)

    Example:
        >>> @pykernel_gen(grid=(2,2))
        >>> def matmul(lhs, rhs, out):
        >>>     lhs_accessor = TensorAccessor(lhs)
        >>>     @datamovement()
        >>>     def dm_reader(...):
        >>>         shard = lhs_cb.reserve()
        >>>         dma(lhs_accessor[idx, 0], shard).wait()
    """

    def __init__(self, tensor: Any):
        """
        Create a TensorAccessor from a tensor argument.

        Args:
            tensor: Tensor that must be a registered top-level argument

        Raises:
            ValueError: If tensor is not a top-level argument
        """
        try:
            self.name = get_tensor_global_name(tensor)
        except ValueError:
            raise ValueError(
                "TensorAccessor must be created from a top level tensor argument"
            )
        self.shape = tensor.shape
        self.dtype = tensor.dtype
