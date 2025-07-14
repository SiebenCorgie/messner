/// Implementation of the EKL dialect ops.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/Ops.h"

#include "messner/Dialect/EKL/IR/Dialect.h"

#include <cstdio>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Typing/Bound.h>
#include <mlir/Typing/Contradiction.h>
#include <mlir/Typing/TypeChecker.h>
#include <optional>

using namespace mlir;
using namespace mlir::ekl;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "messner/Dialect/EKL/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ProgramOp implementation
//===----------------------------------------------------------------------===//

auto ProgramOp::verifyRegions() -> LogicalResult
{
    // All descendants must be declarations.
    for (auto &op : this->getOps())
        if (!op.hasTrait<ekl::OpTrait::Declaration>()) {
            auto diag = emitError("invalid program");
            diag.attachNote(op.getLoc()) << "expected declaration";
            return diag;
        }

    return success();
}

//===----------------------------------------------------------------------===//
// FuncOp implementation
//===----------------------------------------------------------------------===//

void FuncOp::build(
    OpBuilder &,
    OperationState &state,
    StringAttr name,
    FunctionType type)
{
    assert(name && type);

    state.addAttribute(getSymNameAttrName(state.name), name);
    state.addAttribute(
        getFunctionTypeAttrName(state.name),
        TypeAttr::get(type));
    state.addRegion();
}

auto FuncOp::verify() -> LogicalResult
{
    // Can't export functions.
    if (isPublic()) return emitOpError("visibility can't be public");

    return success();
}

auto FuncOp::verifyRegions() -> LogicalResult
{
    if (isExternal()) return success();

    // There must be a YieldOp terminator if the definition is not empty.
    if (getBody()->empty() || !llvm::isa<YieldOp>(&getBody()->back())) {
        auto diag = emitOpError("requires `ekl.yield` terminator");
        if (!getBody()->empty())
            diag.attachNote(getBody()->back().getLoc())
                << "found `" << getBody()->back().getName() << "` instead";
        return diag;
    }

    return success();
}

auto FuncOp::typeCheck(Typing::AbstractTypeChecker &typeChecker)
    -> std::optional<Typing::Contradiction>
{
    std::printf("FnBing\n");
    return std::nullopt;
}

//===----------------------------------------------------------------------===//
// KernelOp implementation
//===----------------------------------------------------------------------===//

static auto getAllArgAttrs(KernelOp op) -> SmallVector<Attribute>
{
    auto allAttrs = llvm::to_vector(
        op.getArgAttrs() ? op.getArgAttrs()->getValue()
                         : ArrayRef<Attribute>{});
    allAttrs.resize(
        op.getBody()->getNumArguments(),
        DictionaryAttr::get(op.getContext()));
    return allAttrs;
}

void KernelOp::build(OpBuilder &, OperationState &state, StringAttr name)
{
    assert(name);

    state.addAttribute(getSymNameAttrName(state.name), name);
    state.addRegion()->emplaceBlock();
}

auto KernelOp::insertArgument(
    unsigned pos,
    ABIType type,
    Location loc,
    DictionaryAttr attrs) -> BlockArgument
{
    assert(pos <= getBody()->getNumArguments());
    assert(type);

    const auto result = getBody()->insertArgument(pos, type, loc);
    if (attrs) setAttrs(result, attrs);
    return result;
}

void KernelOp::eraseArgument(BlockArgument arg)
{
    assert(arg && arg.getOwner() == getBody());

    auto allAttrs = getAllArgAttrs(*this);
    allAttrs.erase(std::next(allAttrs.begin(), arg.getArgNumber()));
    setArgAttrsAttr(TupleAttr::get(getContext(), allAttrs));
}

void KernelOp::setAttrs(BlockArgument arg, DictionaryAttr attrs)
{
    assert(arg && arg.getOwner() == getBody());

    auto allAttrs                = getAllArgAttrs(*this);
    allAttrs[arg.getArgNumber()] = attrs;
    setArgAttrsAttr(TupleAttr::get(getContext(), allAttrs));
}

void KernelOp::setAttr(BlockArgument arg, NamedAttribute attr)
{
    assert(arg && arg.getOwner() == getBody());

    auto allAttrs  = getAllArgAttrs(*this);
    auto &argAttrs = allAttrs[arg.getArgNumber()];
    NamedAttrList dict(llvm::cast<DictionaryAttr>(argAttrs));
    dict.set(attr.getName(), attr.getValue());
    argAttrs = DictionaryAttr::get(getContext(), dict);
}

auto KernelOp::removeAttr(BlockArgument arg, StringRef name) -> Attribute
{
    assert(arg && arg.getOwner() == getBody());

    auto allAttrs  = getAllArgAttrs(*this);
    auto &argAttrs = allAttrs[arg.getArgNumber()];
    NamedAttrList dict(llvm::cast<DictionaryAttr>(argAttrs));
    const auto result = dict.erase(name);
    argAttrs          = DictionaryAttr::get(getContext(), dict);
    return result;
}

auto KernelOp::verify() -> LogicalResult
{
    // If argument attributes are specified, their number must match.
    if (const auto allArgAttrs = getArgAttrs(); allArgAttrs) {
        if (allArgAttrs->size() != getBody()->getNumArguments())
            return emitOpError()
                << "expected " << getBody()->getNumArguments()
                << " argument attributes, but got " << allArgAttrs->size();
    }

    return success();
}

//===----------------------------------------------------------------------===//
// GetStaticOp implementation
//===----------------------------------------------------------------------===//

auto GetStaticOp::verifySymbolUses(SymbolTableCollection &symbolTable)
    -> LogicalResult
{
    auto targetOp =
        symbolTable.lookupNearestSymbolFrom(*this, getTargetNameAttr());
    auto staticOp = llvm::dyn_cast_if_present<StaticOp>(targetOp);
    if (!staticOp) {
        auto diag = emitOpError("'")
                 << getTargetName() << "' does not reference a static variable";
        if (targetOp)
            diag.attachNote(targetOp->getLoc()) << "references this symbol";
        return diag;
    }

    if (getType() != staticOp.getType()) {
        auto diag = emitOpError("expected ")
                 << staticOp.getType() << ", but got " << getType();
        diag.attachNote(targetOp->getLoc()) << "symbol declared here";
        return diag;
    }

    return success();
}

//===----------------------------------------------------------------------===//
// PromoteOp implementation
//===----------------------------------------------------------------------===//

auto PromoteOp::fold(FoldAdaptor) -> OpFoldResult
{
    // TODO: Implement.
    return {};
}

//===----------------------------------------------------------------------===//
// BroadcastOp implementation
//===----------------------------------------------------------------------===//

auto BroadcastOp::fold(FoldAdaptor) -> OpFoldResult
{
    // TODO: Implement.
    return {};
}

//===----------------------------------------------------------------------===//
// CoerceOp implementation
//===----------------------------------------------------------------------===//

auto CoerceOp::fold(FoldAdaptor) -> OpFoldResult
{
    // TODO: Implement.
    return {};
}
/*
//===----------------------------------------------------------------------===//
// Arithmetic operator implementation
//===----------------------------------------------------------------------===//

static std::optional<Typing::Contradiction> typeCheckArithmeticOp(
    Operation *op,
    ::mlir::Typing::AbstractTypeChecker &typeChecker,
    function_ref<uint64_t(ArrayRef<uint64_t>)> combineIndexBounds)
{
    // The operands must all unify or broadcast together.
    ArithmeticType unifiedTy;

    if (auto contra = adaptor.broadcastAndUnify(
            adaptor.getParent()->getOperands(),
            unifiedTy,
            "arithmetic type"))
        return contra;

    // Arithmetic operations need to properly update the upper bounds on the
    // types of index values they produce.
    if (llvm::isa<ekl::IndexType>(unifiedTy.getScalarType())) {
        const auto operandUpperBounds = llvm::to_vector(
            llvm::map_range(
                adaptor.getParent()->getOperands(),
                [&](Value operand) {
                    return llvm::cast<ekl::IndexType>(
                               getScalarType(adaptor.getType(
                                   llvm::cast<Expression>(operand))))
                        .getUpperBound();
                }));
        return adaptor.refineBound(
            llvm::cast<Expression>(adaptor.getParent()->getResult(0)),
            unifiedTy.cloneWith(
                ekl::IndexType::get(
                    unifiedTy.getContext(),
                    combineIndexBounds(operandUpperBounds))));
    }

    // The result type is the unified type.
    return adaptor.refineBound(
        llvm::cast<Expression>(adaptor.getParent()->getResult(0)),
        unifiedTy);
}
*/
//===----------------------------------------------------------------------===//
// MinOp implementation
//===----------------------------------------------------------------------===//

auto MinOp::fold(FoldAdaptor) -> OpFoldResult
{
    // TODO: Implement.
    return {};
}

std::optional<Typing::Contradiction>
MinOp::typeCheck(::mlir::Typing::AbstractTypeChecker &typeChecker)
{

    auto lhs          = getLhs();
    auto rhs          = getRhs();
    auto result       = getResult();
    auto result_bound = typeChecker.get(result);

    // llvm::dbgs() << "Result bound: " << result_bound << "\n";

    // try to meet the bounds of the result for both sides
    auto mlhs = typeChecker.meet(lhs, result.getType());
    auto mrhs = typeChecker.meet(rhs, result.getType());

    // lhs is not within result's bounds
    if (auto maybeContra = mlhs.toContra()) {
        std::printf("Could not meet result + lhs");
        maybeContra->attachNote(getLoc());
        maybeContra->append("here");
        return maybeContra;
    }
    // rhs is not within result's bounds
    if (auto maybeContra = mrhs.toContra()) {
        std::printf("Could not meet result + rhs");
        maybeContra->attachNote(getLoc());
        maybeContra->append(" here");
        return maybeContra;
    }

    // Otherwise we refined _something_
    return std::nullopt;
}

//===----------------------------------------------------------------------===//
// EKLDialect implementation
//===----------------------------------------------------------------------===//

void EKLDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "messner/Dialect/EKL/IR/Ops.cpp.inc"
        >();
}
