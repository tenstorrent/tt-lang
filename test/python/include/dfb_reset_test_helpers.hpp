// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

template <typename... DFBDescriptors>
inline void retain_dfb_liveness() {
  ((void)DFBDescriptors::index, ...);
}
