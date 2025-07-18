/// Implementation of the EKL dialect semantic analysis.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/Analysis/Shape.h"
#include "messner/Dialect/EKL/IR/Ops.h"
#include "messner/Dialect/EKL/IR/TypeSystem.h"
#include "messner/Dialect/EKL/IR/TypeUtils.h"
#include "messner/Dialect/EKL/IR/Types.h"

#include <cstddef>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Typing/Contradiction.h>
#include <mlir/Typing/TypeChecker.h>
#include <optional>

using namespace mlir;
using namespace mlir::Typing;
using namespace mlir::ekl;

//===----------------------------------------------------------------------===//
// FuncOp implementation
//===----------------------------------------------------------------------===//

auto FuncOp::checkSemantics(SmallVectorImpl<Diagnostic> &diagnostics)
    -> LogicalResult
{
    if (!isExternal()) return success();

    // There may be at most one result value.
    if (getResultTypes().size() > 1) {
        diagnostics.emplace_back(getLoc(), DiagnosticSeverity::Error)
            << "FFI functions can have at most one result";
    }

    // All arguments and the result must have an ABIType.
    for (const auto ty : getFunctionType().getInputs()) {
        if (!llvm::isa<ABIType>(ty)) {
            auto &diag =
                diagnostics.emplace_back(getLoc(), DiagnosticSeverity::Error)
                << "FFI function arguments must have ABI types";
            diag.attachNote() << ty << " is not an ABI type";
        }
    }
    if (!getResultTypes().empty() && !llvm::isa<ABIType>(getResultTypes()[0])) {
        auto &diag =
            diagnostics.emplace_back(getLoc(), DiagnosticSeverity::Error)
            << "FFI function results must have ABI types";
        diag.attachNote() << getResultTypes()[0] << " is not an ABI type";
    }

    return success();
}

//===----------------------------------------------------------------------===//
// KernelOp implementation
//===----------------------------------------------------------------------===//

auto KernelOp::checkSemantics(SmallVectorImpl<Diagnostic> &diagnostics)
    -> LogicalResult
{
    // All arguments must have an ABIType.
    for (const auto &arg : getArguments()) {
        if (!llvm::isa<ABIType>(arg.getType())) {
            auto &diag =
                diagnostics.emplace_back(getLoc(), DiagnosticSeverity::Error)
                << "kernel arguments must have ABI types";
            diag.attachNote(arg.getLoc())
                << arg.getType() << " is not an ABI type";
        }
    }

    return success();
}

//===----------------------------------------------------------------------===//
// StaticOp implementation
//===----------------------------------------------------------------------===//

auto StaticOp::checkSemantics(SmallVectorImpl<Diagnostic> &diagnostics)
    -> LogicalResult
{
    // Scalable array references must be imports.
    if (getType().isScalable()) {
        if (!isImported())
            diagnostics.emplace_back(getLoc(), DiagnosticSeverity::Error)
                << "can't define static scalable arrays";
    }

    // Public symbols must have initializers.
    if (isPublic() && isDeclaration())
        diagnostics.emplace_back(getLoc(), DiagnosticSeverity::Error)
            << "can't export a static declaration";

    // The initializer must be assignable to the declared type.
    if (const auto maybeInit = getInitializer(); maybeInit) {
        if (!getTypeSystem(getContext())
                 .isSubtype(
                     maybeInit->getArrayType(),
                     getType().getCellType())) {
            diagnostics.emplace_back(getLoc(), DiagnosticSeverity::Error)
                << "can't initialize a variable of type "
                << getType().getCellType() << " with a value of type "
                << maybeInit->getArrayType();
        }
    }

    return success();
}

//===----------------------------------------------------------------------===//
// YieldOp implementation
//===----------------------------------------------------------------------===//

auto YieldOp::typeCheck(AbstractTypeChecker &typeChecker)
    -> std::optional<Contradiction>
{
    // The YieldOp is allowed to invalidate its parent so that it can adjust its
    // result type based on the result of its functor regions. Bounded execution
    // is guaranteed when no parent deduces block argument types based on its
    // functor result types.
    typeChecker.invalidate((*this)->getParentOp());
    return {};
}

//===----------------------------------------------------------------------===//
// PromoteOp implementation
//===----------------------------------------------------------------------===//

auto PromoteOp::typeCheck(AbstractTypeChecker &tc)
    -> std::optional<Contradiction>
{
    auto input  = getOperand();
    auto tin    = tc.get(input);
    auto output = getResult();
    auto tout   = tc.get(output);

    // Is already equal
    if (tin == tout) return std::nullopt;

    // Try to promote
    auto promotion = tc.getTypeSystem(getOperation()).promote(tin, tout);

    if (promotion == NULL) {
        // failed make this fatal for now
        auto f = tc.fatal(getLoc());
        f << "Can not promote from " << tin << " to " << tout;
        return f;
    }

    return std::nullopt;
}

//===----------------------------------------------------------------------===//
// BroadcastOp implementation
//===----------------------------------------------------------------------===//

auto BroadcastOp::typeCheck(AbstractTypeChecker &tc)
    -> std::optional<Contradiction>
{

    ArrayType inty = llvm::dyn_cast<ArrayType>(getOperand().getType());
    auto inshape   = inty.getShape();
    // Do not broadcast if dimentions don't match.
    //(also the broadcast call panics otherwise).
    if (inshape.size() != getResultShape().size()) {
        auto f = tc.fatal(getLoc());
        f << "Can not broadcast " << inshape.size() << "D shape to "
          << getResultShape().size() << "D";
        return f;
    }

    // Test whether the input shape broadcasts to the output shape
    auto result = broadcast(inshape, getResultShape());

    if (failed(result)) {
        auto f = tc.fatal(getLoc());
        f << "Can not broadcast from" << getOperand().getType() << " to "
          << getResult().getType();
        return f;
    }

    // Update result type to broadcast shape
    ArrayType rty      = llvm::dyn_cast<ArrayType>(getResult().getType());
    auto newResultType = rty.cloneWith(inshape);
    // tell the tc to meet the broadcasted shape
    auto meetResult    = tc.meet(getResult(), newResultType);

    if (auto contra = meetResult.toContra())
        return contra;
    else
        return std::nullopt;
}

//===----------------------------------------------------------------------===//
// CoerceOp implementation
//===----------------------------------------------------------------------===//

auto CoerceOp::typeCheck(AbstractTypeChecker &tc)
    -> std::optional<Contradiction>
{

    auto inty = getOperand().getType();
    if (!inty) {
        // not yet set
        return Contradiction();
    }

    if (mlir::ekl::canCoerce(tc, getOperand(), inty, getType()))
        return std::nullopt;
    {
        auto f = tc.fatal(getLoc());
        f << "Can not coerce form " << inty << " to " << getType();
        return f;
    }
}
