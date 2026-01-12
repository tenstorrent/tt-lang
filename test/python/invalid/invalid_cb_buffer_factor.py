# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: CB buffer_factor must be in range [1, 32].
"""


# CHECK: buffer_factor must be in range [1, 32]
# Validation happens in CircularBuffer.__init__, no ttnn needed
from ttlang import ttl
ttl.make_circular_buffer_like(None, shape=(1, 1), buffer_factor=0)
