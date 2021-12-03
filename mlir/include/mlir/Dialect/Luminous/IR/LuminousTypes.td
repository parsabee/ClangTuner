//===- LuminousTypes.td - Luminous dialect types -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Luminous dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LUMINOUS_IR_ASYNCTYPES
#define MLIR_DIALECT_LUMINOUS_IR_ASYNCTYPES

include "mlir/Dialect/Luminous/IR/LuminousDialect.td"

//===----------------------------------------------------------------------===//
// Luminous types
//===----------------------------------------------------------------------===//

def Luminous_AsyncToken : DialectType<
      Luminous_Dialect,CPred<"$_self.isa<::mlir::luminous::AsyncTokenType>()">,
      "async token type">,
  BuildableType<"mlir::luminous::AsyncTokenType::get($_builder.getContext())">;

#endif // MLIR_DIALECT_LUMINOUS_IR_ASYNCTYPES