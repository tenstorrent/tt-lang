// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace tt::ttl {

/// Registers the BufferizableOpInterface external models used by TTL.
///
/// One-Shot Bufferize only knows about ops that model the interface inside the
/// MLIR context's dialect registry. Since `ttl.attach_cb` keeps operating on
/// both tensors (pre-bufferization) and memrefs (post-bufferization), we hook
/// the aliasing semantics via an external model so that any context that loads
/// the TTL dialect (ttlang-opt, Python bindings, C API, etc.) sees the same
/// bufferization behavior without requiring static registration.
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);

} // namespace tt::ttl
} // namespace mlir

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H
