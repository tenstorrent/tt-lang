# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Compute thread builder for ME2E tests.

Provides builders for compute threads that execute elementwise operations
on data in circular buffers. Uses Python MLIR bindings for type safety.
"""

from typing import Callable, List

from ttmlir.ir import Context, Location, Module

import ttl.dialects.ttl as ttl

from ..config import E2EConfig
from .thread_builder import ThreadBuilder


class ComputeThreadBuilder(ThreadBuilder):
    """
    Compute thread builder - elementwise operations.

    Generates compute thread functions using Python MLIR bindings.
    Supports both single operations and custom/fused operations via callback.
    """

    def build_compute(self, op_str: str, arity: int, num_outputs: int = 1) -> None:
        """
        Build compute thread for a single elementwise operation.

        Args:
            op_str: Operation name (e.g., "add", "exp").
            arity: Number of inputs (1 for unary, 2 for binary).
            num_outputs: Number of outputs (default 1).
        """
        # Set total CB count for base_cta_index calculation.
        self._total_cb_count = arity + num_outputs

        input_cbs = list(range(arity))
        output_cbs = list(range(arity, arity + num_outputs))

        def compute_fn(inputs: List) -> List:
            op_func = getattr(ttl, op_str, None)
            if op_func is None:
                raise ValueError(f"Unknown TTL op: ttl.{op_str}")

            if arity == 1:
                result = op_func(self.tile_tensor_type, inputs[0], loc=self.loc)
            elif arity == 2:
                result = op_func(
                    self.tile_tensor_type, inputs[0], inputs[1], loc=self.loc
                )
            else:
                raise ValueError(f"Unsupported arity: {arity}")

            return [result]

        self._build_compute_thread(
            name=f"compute_{op_str}",
            input_cbs=input_cbs,
            output_cbs=output_cbs,
            compute_fn=compute_fn,
        )

    def build_compute_custom(
        self,
        name: str,
        input_cbs: List[int],
        output_cbs: List[int],
        compute_fn: Callable[[List], List],
    ) -> None:
        """
        Build compute thread with custom/fused operations via callback.

        Use this for fused operations like exp(a + b) or sqrt(abs(a)).

        Args:
            name: Function name.
            input_cbs: List of input CB indices.
            output_cbs: List of output CB indices.
            compute_fn: Callback that takes list of input tensors, returns list of output tensors.
                        The callback receives attached input tensors and should return result tensors.

        Example:
            # For exp(a + b):
            builder.build_compute_custom(
                name="compute_exp_add",
                input_cbs=[0, 1],
                output_cbs=[2],
                compute_fn=lambda inputs: [
                    ttl.exp(tt, ttl.add(tt, inputs[0], inputs[1], loc=loc), loc=loc)
                ],
            )
        """
        # Set total CB count for base_cta_index calculation.
        all_cbs = set(input_cbs + output_cbs)
        self._total_cb_count = max(all_cbs) + 1 if all_cbs else 0

        self._build_compute_thread(
            name=name,
            input_cbs=input_cbs,
            output_cbs=output_cbs,
            compute_fn=compute_fn,
        )
