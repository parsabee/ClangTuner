//
// Created by Parsa Bagheri on 6/16/21.
//

#include "SemaCheck.h"

namespace clang {
namespace tuner {

static bool isAnMLIRAttr(const char *name) {
  return (std::strcmp(name, "mlir::forloop") == 0) ||
      (std::strcmp(name, "mlir::parallel") == 0) ||
      (std::strcmp(name, "mlir::opt") == 0) ||
      (std::strcmp(name, "mlir::collapse") == 0) ||
      (std::strcmp(name, "mlir::omp") == 0);
}

bool isAnMLIRAttr(const Attr *attr) {
  return (isAnMLIRAttr(attr->getNormalizedFullName().c_str()));
}

} // namespace tuner
} // namespace clang
