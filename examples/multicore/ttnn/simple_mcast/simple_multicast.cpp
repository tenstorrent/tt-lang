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
    constexpr uint32_t reduce_core_x = get_compile_time_arg_val(2);
    constexpr uint32_t reduce_core_y = get_compile_time_arg_val(3);
    constexpr uint32_t mcast_start_x = get_compile_time_arg_val(4);
    constexpr uint32_t mcast_start_y = get_compile_time_arg_val(5);
    constexpr uint32_t mcast_end_x = get_compile_time_arg_val(6);
    constexpr uint32_t mcast_end_y = get_compile_time_arg_val(7);
    constexpr uint32_t num_dests = get_compile_time_arg_val(8);

    // Get semaphore addresses
    const uint32_t start_sem_addr = get_semaphore(start_sem_idx);
    const uint32_t done_sem_addr = get_semaphore(done_sem_idx);

    volatile tt_l1_ptr uint32_t* start_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(start_sem_addr);
    volatile tt_l1_ptr uint32_t* done_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_sem_addr);

    const bool is_reduce_core = (core_id == 0);

    // Print physical NOC coordinates
    DPRINT << "Core " << core_id << " started at physical NOC coords (" << (uint32_t)my_x[0] << "," << (uint32_t)my_y[0] << ")" << ENDL();

    // NOC addresses for signaling - use actual physical coordinates
    const uint64_t done_sem_noc_addr = get_noc_addr(reduce_core_x, reduce_core_y, done_sem_addr);

    // For multicast, we need to compute the actual physical coordinate range
    // Since all cores are on the same row (y=2 for all), and x goes from 1 to 4,
    // we can use compile-time args if they were physical, or compute dynamically
    // For now, let's use the passed coordinates but note they should be physical
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

        // Reduce core: multicast start signal to ALL cores (including self with loopback)
        noc_semaphore_set(start_sem_ptr, 1);
        DPRINT << "Core 0: Multicasting start signal to " << num_dests << " other cores" << ENDL();

        // num_dests is the number of OTHER cores (loopback_src includes sender automatically)
        noc_semaphore_set_multicast_loopback_src(start_sem_addr, start_sem_mcast_addr, num_dests);
        noc_async_atomic_barrier();
        DPRINT << "Core 0: Multicast complete" << ENDL();

        // Reduce core also signals itself as done
        noc_semaphore_inc(done_sem_noc_addr, 1);
        noc_async_atomic_barrier();
        DPRINT << "Core 0: Incremented done_sem, waiting for " << (num_dests + 1) << ENDL();

        // Wait for all cores (num_dests + 1 = total cores including self)
        noc_semaphore_wait(done_sem_ptr, num_dests + 1);
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
