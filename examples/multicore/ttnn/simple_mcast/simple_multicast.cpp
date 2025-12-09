// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Simple multicast semaphore example.
 *
 * Core 0 (reduce core) multicasts a start signal to all cores.
 * All cores increment a done semaphore back to core 0.
 * Core 0 waits for all cores to complete.
 */

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // Runtime args
    const uint32_t core_id = get_arg_val<uint32_t>(0);

    // Compile-time args
    constexpr uint32_t start_sem_idx = get_compile_time_arg_val(0);
    constexpr uint32_t done_sem_idx = get_compile_time_arg_val(1);
    constexpr uint32_t num_cores = get_compile_time_arg_val(2);

    // Get semaphore addresses
    const uint32_t start_sem_addr = get_semaphore(start_sem_idx);
    const uint32_t done_sem_addr = get_semaphore(done_sem_idx);

    volatile tt_l1_ptr uint32_t* start_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(start_sem_addr);
    volatile tt_l1_ptr uint32_t* done_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_sem_addr);

    const bool is_reduce_core = (core_id == 0);

    // Get my physical NOC coordinates
    const uint32_t my_phys_x = my_x[0];
    const uint32_t my_phys_y = my_y[0];

    DPRINT << "Core " << core_id << " started at physical NOC coords (" << my_phys_x << "," << my_phys_y << ")" << ENDL();

    // For multicast, compute the range based on number of cores
    // All cores are in a row with same y coordinate, x coordinates are consecutive
    // Multicast from my_phys_x to my_phys_x + (num_cores - 1)
    const uint32_t mcast_start_x = my_phys_x;
    const uint32_t mcast_start_y = my_phys_y;
    const uint32_t mcast_end_x = my_phys_x + (num_cores - 1);
    const uint32_t mcast_end_y = my_phys_y;

    // NOC address for signaling done to reduce core
    // Reduce core is core_id 0. If I'm the reduce core, use my coords. Otherwise, need to know reduce core's coords.
    // Since all cores are in a row, reduce core is at (my_phys_x - core_id, my_phys_y)
    const uint32_t reduce_core_phys_x = my_phys_x - core_id;
    const uint32_t reduce_core_phys_y = my_phys_y;
    const uint64_t done_sem_noc_addr = get_noc_addr(reduce_core_phys_x, reduce_core_phys_y, done_sem_addr);

    DPRINT << "Core " << core_id << ": Reduce core at (" << reduce_core_phys_x << "," << reduce_core_phys_y << ")" << ENDL();
    const uint64_t start_sem_mcast_addr = get_noc_multicast_addr(
        mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, start_sem_addr);

    // Initialize semaphores
    noc_semaphore_set(start_sem_ptr, 0);
    noc_semaphore_set(done_sem_ptr, 0);

    if (is_reduce_core) {
        DPRINT << "Core 0: Reduce core initializing" << ENDL();
        DPRINT << "Core 0: Multicast range (" << mcast_start_x << "," << mcast_start_y << ") to ("
               << mcast_end_x << "," << mcast_end_y << ")" << ENDL();

        // Reset done semaphore before starting
        noc_semaphore_set(done_sem_ptr, 0);

        // Reduce core signals itself as done BEFORE multicasting
        // (to avoid race where workers finish before reduce core increments)
        noc_semaphore_inc(done_sem_noc_addr, 1);
        noc_async_atomic_barrier();
        DPRINT << "Core 0: Incremented done_sem to 1" << ENDL();

        // Multicast start signal to ALL cores (including self with loopback)
        noc_semaphore_set(start_sem_ptr, 1);
        DPRINT << "Core 0: Multicasting start signal to " << (num_cores - 1) << " other cores" << ENDL();

        // num_cores - 1 = number of OTHER cores (loopback_src includes sender automatically)
        noc_semaphore_set_multicast_loopback_src(start_sem_addr, start_sem_mcast_addr, num_cores - 1);
        noc_async_atomic_barrier();
        DPRINT << "Core 0: Multicast complete, waiting for " << num_cores << " total" << ENDL();

        // Wait for all cores (num_cores total)
        noc_semaphore_wait(done_sem_ptr, num_cores);
        DPRINT << "Core 0: All cores done!" << ENDL();
    } else {
        DPRINT << "Core " << core_id << ": Worker waiting for start signal" << ENDL();

        // Worker cores: wait for start signal
        noc_semaphore_wait(start_sem_ptr, 1);
        DPRINT << "Core " << core_id << ": Start signal received" << ENDL();

        // Signal completion back to reduce core
        noc_semaphore_inc(done_sem_noc_addr, 1);
        noc_async_atomic_barrier();
        DPRINT << "Core " << core_id << ": Signaled done" << ENDL();
    }

    DPRINT << "Core " << core_id << " finished" << ENDL();
}
