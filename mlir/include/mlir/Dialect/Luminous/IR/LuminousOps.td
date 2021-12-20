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
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/DLTI/DLTIBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"


//===----------------------------------------------------------------------===//
// Luminous op definitions
//===----------------------------------------------------------------------===//

class Luminous_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Luminous_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Luminous ModuleOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Luminous ModuleEndOp
//===----------------------------------------------------------------------===//

def Luminous_ModuleEndOp
    : Luminous_Op<"module_end", [Terminator, HasParent<"LuminousModuleOp">]> {
  let summary = "A pseudo op that marks the end of a luminous.module.";
  let description = [{This op terminates the only block inside the only region
                          of a `luminous.module`.}];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Luminous FuncOp
//===----------------------------------------------------------------------===//

def Luminous_LuminousFuncOp : Luminous_Op<"func", [
  HasParent<"LuminousModuleOp">, AutomaticAllocationScope, FunctionLike,
  IsolatedFromAbove, Symbol
]> {
  let summary = "Function executable on a Luminous device";

  let description = [{
    ```
        op :: = `luminous.func` symbol-ref-id `(` argument-list `)`
                memory-attribution
                function-attributes?
                region
    ```

    Example:
    ```mlir
      luminous.func @foo(%arg0: index) attributes{qux: "quux"} {
        ...
      }
    ```

    The generic form illustrates the concept
    ```mlir
      "luminous.func"(%arg: index) {sym_name: "foo", qux : "quux"}
        ({^bb0(% arg0: index):
          ...
        }): (index)->()
    ```
  }];

  let regions = (region AnyRegion : $body);
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
                 CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = [{
    // FunctionLike trait needs access to the functions below.
    friend class OpTrait::FunctionLike<LuminousFuncOp>;

    /// Hooks for the input/output type enumeration in FunctionLike .
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }
    unsigned getNumFuncResults() { return getType().getNumResults(); }

    /// Hook for FunctionLike verifier.
    LogicalResult verifyType();

    /// Verifies the body of the function.
    LogicalResult verifyBody();
  }];

  let verifier = [{ return ::verify(*this); }];
  let printer = [{ printLuminousFuncOp(p, *this); }];
  let parser = [{ return parseLuminousFuncOp(parser, result); }];
}

//===----------------------------------------------------------------------===//
// Luminous ReturnOp
//===----------------------------------------------------------------------===//

def Luminous_ReturnOp : Luminous_Op<"return", [HasParent<"LuminousFuncOp">,
                                               NoSideEffect, Terminator]>,
                        Arguments<(ins)>,
                        Results<(outs)> {
  let summary = "Terminator for Luminous functions.";
  let description = [{
    A terminator operation for regions that appear in the body of `luminous.func`.
    `luminous.return` takes no operands.
  }];

  let assemblyFormat = "attr-dict";
}

#endif // LUMINOUS_OPS