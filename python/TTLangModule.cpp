#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

// Forward declare registration function from generated code
namespace mlir::tt::ttlang {
void registerPasses();
}

NB_MODULE(_ttlang, m) {
  m.doc() = "tt-lang Python bindings";

  // Register tt-lang passes
  mlir::tt::ttlang::registerPasses();
}
