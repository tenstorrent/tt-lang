// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_ANALYSIS_PLANNINGRESULT_H
#define TTLANG_ANALYSIS_PLANNINGRESULT_H

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"

#include <cassert>
#include <string>
#include <utility>
#include <variant>

namespace mlir::tt {

/// Diagnostic returned by read-only planning for malformed input IR.
struct PlanningDiagnostic {
  PlanningDiagnostic(Operation *operation, std::string message,
                     Operation *noteOperation = nullptr, std::string note = {})
      : operation(operation), message(std::move(message)),
        noteOperation(noteOperation), note(std::move(note)) {
    assert(operation && "invalid IR requires a diagnostic anchor");
    assert(!this->message.empty() &&
           "invalid IR requires a diagnostic message");
    assert((noteOperation == nullptr) == this->note.empty() &&
           "a note requires an anchor and text");
  }

  PlanningDiagnostic() = delete;

  /// Operation to which the pass driver attaches the diagnostic.
  Operation *operation;

  /// Diagnostic text without an operation-name prefix.
  std::string message;

  /// Optional second operation the diagnostic refers to, with its note text.
  Operation *noteOperation;
  std::string note;
};

/// Emits `diagnostic` as an error on its operation, with its note when set.
inline InFlightDiagnostic
emitPlanningDiagnostic(const PlanningDiagnostic &diagnostic) {
  InFlightDiagnostic error =
      diagnostic.operation->emitError(diagnostic.message);
  if (diagnostic.noteOperation) {
    error.attachNote(diagnostic.noteOperation->getLoc()) << diagnostic.note;
  }
  return error;
}

/// Result of constructing immutable analysis facts or a rewrite plan.
///
/// A planned result satisfies the producer's contract, a rejected result
/// identifies a legal but inapplicable candidate, and an invalid-IR result
/// requires the pass driver to diagnose the input before mutation. `RejectionT`
/// gives each producer a typed reason without coupling the common result
/// contract to stage-specific policy.
template <typename PlanT, typename RejectionT = std::monostate>
class PlanningResult {
public:
  static PlanningResult planned(PlanT plan) {
    return PlanningResult(std::in_place_index<0>, std::move(plan));
  }

  static PlanningResult rejected(RejectionT rejection) {
    return PlanningResult(std::in_place_index<1>, std::move(rejection));
  }

  static PlanningResult invalidIR(Operation *operation, std::string message,
                                  Operation *noteOperation = nullptr,
                                  std::string note = {}) {
    return PlanningResult(std::in_place_index<2>,
                          PlanningDiagnostic{operation, std::move(message),
                                             noteOperation, std::move(note)});
  }

  bool isPlanned() const { return storage.index() == 0; }
  bool isRejected() const { return storage.index() == 1; }
  bool isInvalidIR() const { return storage.index() == 2; }

  const PlanT &getPlan() const {
    assert(isPlanned() && "planning result does not contain a plan");
    return std::get<0>(storage);
  }

  PlanT takePlan() && {
    assert(isPlanned() && "planning result does not contain a plan");
    return std::move(std::get<0>(storage));
  }

  const RejectionT &getRejection() const {
    assert(isRejected() && "planning result does not contain a rejection");
    return std::get<1>(storage);
  }

  RejectionT takeRejection() && {
    assert(isRejected() && "planning result does not contain a rejection");
    return std::move(std::get<1>(storage));
  }

  const PlanningDiagnostic &getInvalidIR() const {
    assert(isInvalidIR() && "planning result does not contain a diagnostic");
    return std::get<2>(storage);
  }

private:
  template <std::size_t Index, typename ValueT>
  PlanningResult(std::in_place_index_t<Index> index, ValueT &&value)
      : storage(index, std::forward<ValueT>(value)) {}

  std::variant<PlanT, RejectionT, PlanningDiagnostic> storage;
};

} // namespace mlir::tt

#endif // TTLANG_ANALYSIS_PLANNINGRESULT_H
