//===-- OpenMPOps.td - OpenMP dialect operation definitions *- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the OpenMP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LUMINOUS_OPS
#define LUMINOUS_OPS

include "mlir/Dialect/Luminous/IR/LuminousTypes.td"
include "mlir/Dialect/DLTI/DLTIBase.td"
include "mlir/Dialect/Luminous/IR/LuminousDialect.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"


//===----------------------------------------------------------------------===//
// Luminous op definitions
//===----------------------------------------------------------------------===//

class Luminous_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Luminous_Dialect, mnemonic, traits>;

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

def Luminous_WaitOp : Luminous_Op<"wait"> {
  let summary = "Wait for async ops to complete.";
  let description = [{
    Blocks until the list of provided `async_token`s are complete. If
    the `async` keyword is available, the op generates an `async_token` which
    merely serves as a shortcut for the list of tokens that the op is waiting
    on.
    This op behaves similar to `gpu.wait` op. For more info, refer to gpu.wait

    op ::= `luminous.wait` `async`? (`[` argument-list `]`)?
            (`->` `!luminous.async_token`)?
  }];

  let arguments = (ins Variadic<Luminous_AsyncTokenType> : $asyncDependencies);
  let results = (outs Optional<Luminous_AsyncTokenType>:$asyncToken);

  // Custom parser for async dependencies; if `async' is present, constructs
  // an AsyncTokenType.
  let assemblyFormat = [{
      custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
      attr-dict
  }];
}

#endif // LUMINOUS_OPS