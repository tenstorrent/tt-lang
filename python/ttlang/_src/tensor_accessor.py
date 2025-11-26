# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TensorAccessor for indexed tile-level access to tensor arguments in kernels."""

from typing import Optional, Any

from ..constants import SUPPORTED_MEMORY_SPACES


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
        memory_space: Memory space for the accessor ("L1" or "DRAM"), or None for kernel default

    Example:
        >>> @pykernel_gen(grid=(2,2))
        >>> def matmul(lhs, rhs, out):
        >>>     lhs_accessor = TensorAccessor(lhs)
        >>>     @datamovement()
        >>>     def dm_reader(...):
        >>>         shard = lhs_cb.reserve()
        >>>         dma(lhs_accessor[idx, 0], shard).wait()
    """

    def __init__(self, tensor: Any, memory_space: Optional[str] = None):
        """
        Create a TensorAccessor from a tensor argument.

        Args:
            tensor: PyTorch tensor that must have a _global_name attribute
            memory_space: Optional memory space ("L1" or "DRAM"). If None, uses L1.

        Raises:
            ValueError: If tensor is not a top-level argument or memory_space is invalid
        """
        if not hasattr(tensor, "_global_name"):
            raise ValueError(
                "TensorAccessor must be created from a top level tensor argument"
            )
        if memory_space is not None and memory_space not in SUPPORTED_MEMORY_SPACES:
            raise ValueError(
                f"Invalid memory_space: {memory_space!r}. "
                f"Must be one of: {', '.join(sorted(SUPPORTED_MEMORY_SPACES))}"
            )
        self.name = tensor._global_name
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.memory_space = memory_space
