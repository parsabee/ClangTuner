//
// Created by Parsa Bagheri on 6/16/21.
//

#include "SemaCheck.h"

namespace clang {
namespace tuner {

bool isATuneAttr(const AttributedStmt *attrStmt) {
  const auto attrs = attrStmt->getAttrs();
  for (auto attr : attrs) {
    if (isATuneAttr(attr->getSpelling()))
      return true;
  }
  return false;
}

bool isATuneAttr(const char *name) {
  return (std::strcmp(name, "block_dim") == 0);
}

} // namespace tuner
} // namespace clang
