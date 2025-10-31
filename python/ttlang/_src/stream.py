# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Stream wrapper for asynchronous data movement in kernels."""

from typing import Optional, Any


class Stream:
    """
    Wraps a tensor argument for streaming data movement between DRAM and L1.

    Streams enable asynchronous DMA operations in kernel code. The tensor must be
    a top-level function argument (not a local variable).

    Attributes:
        name: Global name of the wrapped tensor
        shape: Shape of the tensor
        dtype: Data type of the tensor (e.g., torch.float32)
        num_buffers: Number of multi-buffering slots (currently unused)
    """

    def __init__(self, tensor: Any, num_buffers: Optional[int] = None):
        """
        Create a stream from a tensor argument.

        Args:
            tensor: PyTorch tensor that must have a _global_name attribute
            num_buffers: Number of buffers for multi-buffering (not yet supported)

        Raises:
            AssertionError: If tensor is not a top-level argument or if num_buffers is set
        """
        if not hasattr(tensor, "_global_name"):
            raise ValueError("Stream must be created from a top level tensor argument")
        self.name = tensor._global_name
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        if num_buffers is not None:
            raise NotImplementedError(
                "Multi-buffering (num_buffers) is not yet supported"
            )
        self.num_buffers = num_buffers
