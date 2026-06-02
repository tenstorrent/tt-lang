# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Composable @ttl.atom op library: reusable kernels and topology builders.

Submodules are imported explicitly, e.g. ``from ttl.ops.mcast import
mcast_rows``. Topology builders return lists of ttl.Pipe to wrap in
ttl.PipeNet inside an atom body; compute ops are @ttl.atom functions meant to
be inlined into a composing atom.
"""
