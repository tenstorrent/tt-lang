// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

template <typename Destination, typename SourceAccessor>
inline void tensor_accessor_read(SourceAccessor source) {
  static_assert(Destination::page_size_bytes > 0);
  CircularBuffer destination(get_compile_time_arg_val(Destination::index));
  Noc noc0(0);
  destination.reserve_back(Destination::pages_per_block);
  noc0.async_read(source, CoreLocalMem<uint32_t>(destination.get_write_ptr()),
                  source.get_aligned_page_size(), {.page_id = 0}, {});
  noc0.async_read_barrier();
  destination.push_back(Destination::pages_per_block);
}

template <typename Destination, uint32_t PageId, typename SourceAccessor>
inline void tensor_accessor_read_page(SourceAccessor source) {
  static_assert(Destination::pages_per_block == 1);
  CircularBuffer destination(get_compile_time_arg_val(Destination::index));
  Noc noc0(0);
  destination.reserve_back(1);
  noc0.async_read(source, CoreLocalMem<uint32_t>(destination.get_write_ptr()),
                  source.get_aligned_page_size(), {.page_id = PageId}, {});
  noc0.async_read_barrier();
  destination.push_back(1);
}

template <typename FirstDestination, typename SecondDestination,
          typename FirstSourceAccessor, typename SecondSourceAccessor>
inline void tensor_accessor_pair_read(FirstSourceAccessor firstSource,
                                      SecondSourceAccessor secondSource) {
  CircularBuffer firstDestination(
      get_compile_time_arg_val(FirstDestination::index));
  CircularBuffer secondDestination(
      get_compile_time_arg_val(SecondDestination::index));
  Noc noc0(0);
  firstDestination.reserve_back(FirstDestination::pages_per_block);
  secondDestination.reserve_back(SecondDestination::pages_per_block);
  noc0.async_read(firstSource,
                  CoreLocalMem<uint32_t>(firstDestination.get_write_ptr()),
                  firstSource.get_aligned_page_size(), {.page_id = 0}, {});
  noc0.async_read(secondSource,
                  CoreLocalMem<uint32_t>(secondDestination.get_write_ptr()),
                  secondSource.get_aligned_page_size(), {.page_id = 0}, {});
  noc0.async_read_barrier();
  firstDestination.push_back(FirstDestination::pages_per_block);
  secondDestination.push_back(SecondDestination::pages_per_block);
}
