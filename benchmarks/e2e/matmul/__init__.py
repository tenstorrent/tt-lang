# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Matmul e2e benchmark (ttl.ops.matmul ksplit vs ttnn.matmul)."""

from benchmarks.e2e.matmul.bench import SPEC

__all__ = ["SPEC"]
