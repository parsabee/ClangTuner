//
// Created by parsa on 5/9/21.
//

#ifndef MLIR_TEST_DIALECT_H
#define MLIR_TEST_DIALECT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/// Include the auto-generated header file containing the declaration of the ClangTune
/// dialect.
#include "ClangTune/Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// ClangTune operations.
#define GET_OP_CLASSES
#include "ClangTune/ClangTune.h.inc"

#endif //MLIR_TEST_DIALECT_H
