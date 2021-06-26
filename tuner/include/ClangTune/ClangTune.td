#ifndef CLANGTUNE
#define CLANGTUNE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// ClangTune Dialect
//===----------------------------------------------------------------------===//

def ClangTune_Dialect : Dialect {
  let name = "clang_tune";

  let summary = "A high-level dialect for tuning";

  let description = [{
    A high-level dialect for tuning.
  }];

  let cppNamespace = "::mlir::clang_tune";
}

class ClangTune_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<ClangTune_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// ClangTune Operations
//===----------------------------------------------------------------------===//

def ConstantOp : ClangTune_Op<"constant", [NoSideEffect]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  // Specify a parser and printer method.
  let parser = [{ return ::parseConstantOp(parser, result); }];
  let printer = [{ return ::print(p, *this); }];

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<(ins "double":$value)>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<(ins "float":$value)>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<(ins "int":$value)>
  ];

  // Invoke a static verify method to verify this constant operation.
  let verifier = [{ return ::verify(*this); }];
}

def TensorAddOp : ClangTune_Op<"tensor_add"> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Specify a parser and printer method.
  let parser = [{ return ::parseBinaryOp(parser, result); }];
  let printer = [{ return ::printBinaryOp(p, *this); }];

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def TensorMulOp : ClangTune_Op<"tensor_mul"> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Specify a parser and printer method.
  let parser = [{ return ::parseBinaryOp(parser, result); }];
  let printer = [{ return ::printBinaryOp(p, *this); }];

  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def ForLoopOp : ClangTune_Op<"for_loop"> {
  let summary = "traditional for loop";
    let description = [{
      traditional for loop
    }];

  let arguments = (ins Variadic<AnyType>);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "int64_t":$lowerBound, "int64_t":$upperBound,
      CArg<"int64_t", "1">:$step, CArg<"ValueRange", "llvm::None">:$iterArgs,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
                 "nullptr">:$bodyBuilder)>,
                 
    OpBuilder<(ins "ValueRange":$lbOperands, "ValueRange":$ubOperands,
          CArg<"int64_t", "1">:$step, CArg<"ValueRange", "llvm::None">:$iterArgs,
          CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
                     "nullptr">:$bodyBuilder)>
  ];

  let extraClassDeclaration = [{
    static StringRef getStepAttrName() { return "step"; }
    static StringRef getLowerBoundAttrName() { return "lower_bound"; }
    static StringRef getUpperBoundAttrName() { return "upper_bound"; }
  }];
}

#endif // CLANGTUNE