//
// Created by parsa on 5/9/21.
//

#include "ClangTune/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::clang_tune;

//===----------------------------------------------------------------------===//
// ClangTuneDialect
//===----------------------------------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void ClangTuneDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ClangTune/ClangTune.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ClangTune Operations
//===----------------------------------------------------------------------===//

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.

//===----------------------------------------------------------------------===//
// ConstantOp

/// A generalized printer for binary operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << op->getName() << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  // If all of the types are the same, print the type directly.
  Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }

  // Otherwise, print a functional type.
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  SmallVector<mlir::OpAsmParser::OperandType, 2> operands;
  llvm::SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (FunctionType funcType = type.dyn_cast<FunctionType>()) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                               result.operands))
      return mlir::failure();
    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

/// The 'OpAsmPrinter' class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
static void print(mlir::OpAsmPrinter &printer, ConstantOp op) {
  printer << "clang_tune.constant ";
  printer.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << op.value();
}

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
static mlir::ParseResult parseConstantOp(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

/// Verifier for the constant operation. This corresponds to the `::verify(...)`
/// in the op definition.
static mlir::LogicalResult verify(ConstantOp op) {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType = op.getResult().getType().dyn_cast<mlir::RankedTensorType>();
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = op.value().getType().cast<mlir::TensorType>();
  if (attrType.getRank() != resultType.getRank()) {
    return op.emitOpError(
               "return type must match the one of the attached value "
               "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return op.emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AddOp

void TensorAddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// MulOp

void TensorMulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// ForLoopOp

using LoopBodyBuilderFn =
    function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

static void createForLoopBody(::mlir::OpBuilder &odsBuilder,
                              ::mlir::OperationState &odsState,
                              ValueRange iterArgs,
                              LoopBodyBuilderFn bodyBuilder) {
  // Create a region and a block for the body.  The argument of the region is
  // the loop induction variable.
  Region *bodyRegion = odsState.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  Value inductionVar = bodyBlock.addArgument(odsBuilder.getIndexType());
  for (Value val : iterArgs)
    bodyBlock.addArgument(val.getType());

  // Create the default terminator if the builder is not provided and if the
  // iteration arguments are not provided. Otherwise, leave this to the caller
  // because we don't know which values to return from the loop.
  if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(odsBuilder);
    odsBuilder.setInsertionPointToStart(&bodyBlock);
    bodyBuilder(odsBuilder, odsState.location, inductionVar,
                bodyBlock.getArguments().drop_front());
  }
}

void ForLoopOp::build(::mlir::OpBuilder &odsBuilder,
                      ::mlir::OperationState &odsState, ValueRange lbOperands,
                      ValueRange ubOperands, int64_t step, ValueRange iterArgs,
                      LoopBodyBuilderFn bodyBuilder) {
  for (Value val : iterArgs)
    odsState.addTypes(val.getType());

  odsState.addAttribute(
      getStepAttrName(),
      odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), step));

  // Add an attribute for the step.
  odsState.addAttribute(
      getStepAttrName(),
      odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), step));

  // Add the lower bound.
  odsState.addOperands(lbOperands);

  // Add the upper bound.
  odsState.addOperands(ubOperands);

  odsState.addOperands(iterArgs);
  createForLoopBody(odsBuilder, odsState, iterArgs, bodyBuilder);
}

void ForLoopOp::build(
    ::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
    int64_t lowerBound, int64_t upperBound, int64_t step, ValueRange iterArgs,
    function_ref<void(OpBuilder &, Location, Value, ValueRange)> bodyBuilder) {
  // Add the lower bound.
  odsState.addAttribute(
      getLowerBoundAttrName(),
      odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), lowerBound));

  // Add the upper bound.
  odsState.addAttribute(
      getUpperBoundAttrName(),
      odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), upperBound));

  odsState.addAttribute(
      getStepAttrName(),
      odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), step));

  odsState.addOperands(iterArgs);
  createForLoopBody(odsBuilder, odsState, iterArgs, bodyBuilder);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "ClangTune/ClangTune.cpp.inc"