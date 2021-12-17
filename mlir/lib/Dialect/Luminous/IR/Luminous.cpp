//===- Luminous.cpp - MLIR Luminous Operations ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Luminous/IR/LuminousDialect.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::luminous;

#include "mlir/Dialect/Luminous/IR/LuminousOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// LuminousDialect
//===----------------------------------------------------------------------===//

void LuminousDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Luminous/IR/LuminousOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// LuminousModuleOp
//===----------------------------------------------------------------------===//

void LuminousModuleOp::build(OpBuilder &builder, OperationState &result,
                             StringRef name) {
  ensureTerminator(*result.addRegion(), builder, result.location);
  result.attributes.push_back(builder.getNamedAttr(
      ::mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name)));
}

static ParseResult parseLuminousModuleOp(OpAsmParser &parser,
                                         OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // If module attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse the module body.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, None, None))
    return failure();

  // Ensure that this module has a valid terminator,
  // The ` SingleBlockImplicitTerminator<"ModuleEndOp"> ' trait of module op
  // provides following routine for that
  LuminousModuleOp::ensureTerminator(*body, parser.getBuilder(),
                                     result.location);
  return success();
}

static void print(OpAsmPrinter &p, LuminousModuleOp op) {
  p << ' ';
  p.printSymbolName(op.getName());
  p.printOptionalAttrDictWithKeyword(op->getAttrs(),
                                     {SymbolTable::getSymbolAttrName()});
  p.printRegion(op->getRegion(0), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Luminous/IR/LuminousOps.cpp.inc"