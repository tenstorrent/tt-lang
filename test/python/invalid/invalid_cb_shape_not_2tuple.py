# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: CB shape must be a 2-tuple.
"""

from ttlang import make_circular_buffer_like

# CHECK: shape must be a 2-tuple
# Validation happens in CircularBuffer.__init__, no ttnn needed
make_circular_buffer_like(None, shape=(1, 1, 1), buffer_factor=2)
