# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: DFB block_count must be in range [1, 32].
"""

# CHECK: block_count must be in range [1, 32]
# Validation happens in CircularBuffer.__init__, no ttnn needed
import ttl

ttl.make_dataflow_buffer_like(None, shape=(1, 1), block_count=0)
