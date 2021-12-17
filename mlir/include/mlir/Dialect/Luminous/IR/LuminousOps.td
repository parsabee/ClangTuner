//===-- LuminousOps.td - Luminous operation definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the Luminous dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LUMINOUS_OPS
#define LUMINOUS_OPS

include "mlir/Dialect/Async/IR/AsyncTypes.td"
include "mlir/Dialect/Luminous/IR/LuminousDialect.td"
include "mlir/Dialect/DLTI/DLTIBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"


//===----------------------------------------------------------------------===//
// Luminous op definitions
//===----------------------------------------------------------------------===//

class Luminous_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Luminous_Dialect, mnemonic, traits>;

/// TODO: I think we eventually will need to drop the HasDefaultDLTIDataLayout
/// when we figure out our devices data layout
def Luminous_LuminousModuleOp : Luminous_Op<"module", [
  DataLayoutOpInterface, HasDefaultDLTIDataLayout, IsolatedFromAbove,
  SymbolTable, Symbol, SingleBlockImplicitTerminator<"ModuleEndOp">
]> {
  let summary = "A top level compilation unit containing code to be run on a "
                "luminous device.";
  let description = [{
    ModuleOp contains routines that are intended to be run on a luminous device.

    ```
    luminous.module @symbol_name {
      luminous.func{}
      ...
      luminous.module_end
    }
    ```
  }];
  let builders = [OpBuilder<(ins "StringRef" : $name)>];
  let parser = [{ return ::parseLuminousModuleOp(parser, result); }];
  let printer = [{ return ::print(p, *this); }];
  let regions = (region SizedRegion<1> : $body);

  // We need to ensure the block inside the region is properly terminated;
  // the auto-generated builders do not guarantee that.
  let skipDefaultBuilders = 1;
}

def Luminous_ModuleEndOp
    : Luminous_Op<"module_end", [Terminator, HasParent<"LuminousModuleOp">]> {
  let summary = "A pseudo op that marks the end of a luminous.module.";
  let description = [{This op terminates the only block inside the only region
                          of a `luminous.module`.}];

  let assemblyFormat = "attr-dict";
}

#endif // LUMINOUS_OPS