#include "mlir/Bindings/Python/NanobindAdaptor.h"
#include "ttlang/Transforms/Passes.h"

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptor;

NB_MODULE(_ttlang, m) {
  m.doc() = "tt-lang Python bindings";

  // Register tt-lang passes
  mlir::tt::ttlang::registerPasses();
}
