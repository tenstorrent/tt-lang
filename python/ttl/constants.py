# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Constants used throughout the DSL."""

DEFAULT_TILE_SIZE = 32
SUPPORTED_MEMORY_SPACES = frozenset(["L1", "DRAM"])
# TT kernel hardware semaphore id capacity. Mirrors kMaxHardwareSemaphoreIds in
# include/ttlang/Dialect/TTL/IR/TTL.h.
MAX_HARDWARE_SEMAPHORE_IDS: int = 16

# Per-core static CB region budget (bytes) when IR has no system descriptor:
# Wormhole and Blackhole total L1 (1464 KiB) minus reserved kernel space (128 KiB)
# per tt-metal dev_mem_map. Matches ChipDesc usable L1 for those architectures
# and ttl-validate-cb-budget fallback; with a device + system_desc, the compiler
# uses ChipDescAttr::getUsableL1Size() instead.
# keep in sync with sim.context_types.DEFAULT_MAX_L1_BYTES
DEFAULT_L1_CB_BUDGET_BYTES: int = 1432 * 1024
