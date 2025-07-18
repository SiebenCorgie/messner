/// Implementation of the EKL dialect type utilities.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/TypeUtils.h"

#include "messner/Dialect/EKL/IR/TypeSystem.h"
#include "messner/Dialect/EKL/IR/Types.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/TypeSystem.h>
#include <mlir/Typing/TypeChecker.h>

using namespace mlir;
using namespace mlir::ekl;

//===----------------------------------------------------------------------===//
// hasConcreteType
//===----------------------------------------------------------------------===//

auto mlir::ekl::hasConcreteType(Operation *op) -> bool
{
    assert(op);

    if (!hasConcreteType(op->getOperandTypes())
        || !hasConcreteType(op->getResultTypes()))
        return false;

    for (auto &region : op->getRegions()) {
        for (auto &block : region)
            if (!hasConcreteType(block.getArgumentTypes())) return false;
    }

    return true;
}

//===----------------------------------------------------------------------===//
// Coersion
//===----------------------------------------------------------------------===//

bool mlir::ekl::canCoerce(
    Typing::AbstractTypeChecker &tc,
    Value owner,
    Type from,
    Type to)
{
    // Upcasting (unifcation) is trivial.
    if (tc.getTypeSystem(owner).isSubtype(from, to)) return true;

    // Can coerce any number type to any other number type.
    if (llvm::isa<NumberType>(from) && llvm::isa<NumberType>(to)) return true;

    // Arrays can be coerced if they have the same shape and coercible scalar
    // types.
    const auto fromArray = llvm::dyn_cast_if_present<ArrayType>(from);
    const auto toArray   = llvm::dyn_cast<ArrayType>(to);
    if (fromArray && toArray) {
        return fromArray.getShape() == toArray.getShape()
            && canCoerce(
                   tc,
                   owner,
                   fromArray.getScalarType(),
                   toArray.getScalarType());
    }

    return false;
}
