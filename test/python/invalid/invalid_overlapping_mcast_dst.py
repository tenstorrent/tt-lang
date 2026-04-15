# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: PipeNet rejects overlapping multicast destinations.

Multicast pipes in a PipeNet share a semaphore pair. The handshake protocol
uses noc_semaphore_set_multicast (overwrite, not atomic increment), so a core
receiving from multiple multicast sources would see corrupted semaphore values.

See: https://github.com/tenstorrent/tt-lang/issues/505
"""

import ttl

# Two multicast pipes both targeting core (0, 0):
#   pipe 0: src=(1, 0) -> dst=(0, 0)-(0, 1)  (covers core (0,0))
#   pipe 1: src=(2, 0) -> dst=(0, 0)-(0, 1)  (covers core (0,0))

# CHECK: PipeNet has overlapping multicast destinations
ttl.PipeNet(
    [
        ttl.Pipe(src=(1, 0), dst=(0, slice(0, 2))),
        ttl.Pipe(src=(2, 0), dst=(0, slice(0, 2))),
    ]
)
