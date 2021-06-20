//
// Created by Parsa Bagheri on 6/16/21.
//

#ifndef TUNER__SEMACHECK_H
#define TUNER__SEMACHECK_H

#include "clang/AST/ASTDumper.h"

namespace clang {
namespace tuner {

bool isATuneAttr(const AttributedStmt *);
bool isATuneAttr(const char *name);

} // namespace tuner
} // namespace clang

#endif // TUNER__SEMACHECK_H
