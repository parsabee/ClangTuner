//
// Created by Parsa Bagheri on 6/16/21.
//

#include "SemaCheck.h"

namespace clang {
namespace tuner {

static bool isATuneAttr(const char *name) {
  return (std::strcmp(name, "parallel_for::mlir_opt") == 0);
}

bool isATuneAttr(const Attr *attr) {
  return (isATuneAttr(attr->getNormalizedFullName().c_str()));
}

} // namespace tuner
} // namespace clang
