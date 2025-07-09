/// Declares the TypeCheckingAdaptor.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/Analysis/Casting.h"
#include "mlir/Typing/TypeCheckOpInterface.h"
#include "mlir/Typing/TypeChecker.h"

#include <mlir/Typing/Bound.h>
#include <mlir/Typing/Contradiction.h>
#include <optional>

namespace mlir::ekl {

bool isFullyTyped(Operation *op)
{
    const auto isUnbounded  = [](Type type) { return !getTypeBound(type); };
    const auto hasUnbounded = [&](TypeRange types) {
        return llvm::any_of(types, isUnbounded);
    };

    if (hasUnbounded(op->getOperandTypes())) return false;
    if (hasUnbounded(op->getResultTypes())) return false;

    return true;
}

/// Type of a value that supports type checking.
using Expression = TypedValue<ExpressionType>;

/// Provides an adaptor around an AbstractTypeChecker bound to some
/// TypeCheckOpInterface.
///
/// The adaptor defines some convenience methods to perform common type checking
/// tasks on an operation, which automatically generate error diagnostics when
/// necessary.
struct TypeCheckingAdaptor : Typing::MLIRTypeChecker {
    /// Initializes a TypeCheckingAdaptor using @p impl for @p parent .
    explicit TypeCheckingAdaptor(
        Typing::MLIRTypeChecker &impl,
        TypeCheckOpInterface parent)
            : m_impl(impl),
              m_parent(parent)
    {}

    /// Gets the AbstractTypeChecker.
    [[nodiscard]] Typing::MLIRTypeChecker &getImpl() { return m_impl; }
    /// Gets the AbstractTypeChecker.
    [[nodiscard]] const Typing::MLIRTypeChecker &getImpl() const
    {
        return m_impl;
    }
    /// Gets the parent operation
    [[nodiscard]] TypeCheckOpInterface getParent() const { return m_parent; }

    /// @copydoc AbstractTypeChecker::getType(Expression)
    [[nodiscard]] virtual Type getType(Value value) const
    {
        return getImpl().get(value);
    }
    /// @copydoc AbstractTypeChecker::refineBound(Expression, Type)
    virtual LogicalResult refineBound(Operation *op, Type incoming)
    {
        return getImpl().refineBound(op, incoming);
    }
    /// @copydoc AbstractTypeChecker::meetBound(Expression)
    virtual LogicalResult meetBound(Value value, Type incoming)
    {
        return getImpl().meetLog(value, Typing::Bound(incoming));
    }
    /// @copydoc AbstractTypeChecker::invalidate(Operation *)
    virtual void invalidate(Operation *op) override
    {
        return getImpl().invalidate(op);
    }

    // TODO: Document all these.

    template<type_constraint ResultType>
    std::optional<Typing::Contradiction>
    require(Type type, ResultType &result, const llvm::Twine &what) const
    {
        if (!type) {
            result = ResultType{};
            return std::nullopt;
        }
        if ((result = llvm::dyn_cast<ResultType>(type))) return std::nullopt;
        return m_impl.fatal(Typing::Source())
            //.explain(type)
            ;
    }

    template<type_constraint ResultType>
    std::optional<Typing::Contradiction>
    require(Expression expr, ResultType &result, const llvm::Twine &what) const
    {
        return require(getType(expr), result, what).explain(expr);
    }

    std::optional<Typing::Contradiction>
    require(Type result, Type supertype) const
    {
        if (!result) return std::nullopt;
        if (m_impl.getTypeSystem(&result.getDialect())
                .isSubtype(result, supertype))
            return std::nullopt;
        // FIXME
        //  return Contradiction(
        //      emitError() << result << " is not a subtype of " << supertype);
        return m_impl.fatal(Typing::Source());
    }

    std::optional<Typing::Contradiction>
    require(Operation *op, Type supertype, Type &result) const
    {
        assert(op->getNumResults() == 1);
        result = m_impl.get(op->getResult(0));
        return require(result, supertype)
            // FIXME
            //.explain(expr)
            ;
    }

    std::optional<Typing::Contradiction>
    unify(ArrayRef<Type> types, Type &result) const;

    std::optional<Typing::Contradiction>
    unify(ValueRange exprs, Type &result) const;

    template<type_constraint ResultType>
    std::optional<Typing::Contradiction>
    unify(ArrayRef<Type> types, ResultType &result, const llvm::Twine &what)
        const
    {
        Type unified;
        if (auto contra = unify(types, unified)) return contra;
        return require(unified, result, what)
            .explainImpl([&](Diagnostic &diag) {
                diag << "after unifying to " << unified;
            })
            .explain(types);
    }

    template<type_constraint ResultType>
    std::optional<Typing::Contradiction>
    unify(ValueRange exprs, ResultType &result, const llvm::Twine &what) const
    {
        const auto types = exprs.getTypes();
        return unify<ResultType>(types, result, what).explain(exprs);
    }

    std::optional<Typing::Contradiction>
    broadcast(ArrayRef<Type> types, SmallVectorImpl<extent_t> &extents) const;

    std::optional<Typing::Contradiction>
    broadcast(ValueRange exprs, SmallVectorImpl<extent_t> &extents) const;

    std::optional<Typing::Contradiction>
    broadcast(Type type, ExtentRange extents, ArrayType &result) const;

    std::optional<Typing::Contradiction>
    broadcast(Expression expr, ExtentRange extents, ArrayType &result) const
    {
        return broadcast(getType(expr), extents, result)
            // FIXME: .explain(expr)
            ;
    }

    std::optional<Typing::Contradiction>
    broadcast(MutableArrayRef<Type> types) const;

    std::optional<Typing::Contradiction>
    broadcast(ValueRange exprs, SmallVectorImpl<Type> &result) const;

    std::optional<Typing::Contradiction> coerce(Type type, Type to) const
    {
        // FIXME: if (!type) return Contradiction::indeterminate();
        if (!type) return std::nullopt;
        if (ekl::canCoerce(type, to)) return std::nullopt;
        // FIXME: return emitError() << "can't coerce " << type << " to " << to;
        return m_impl.fatal(Typing::Source());
    }

    std::optional<Typing::Contradiction> coerce(Expression expr, Type to) const
    {
        return coerce(getType(expr), to)
            // FIXME: .explain(expr)
            ;
    }

    template<broadcast_type_constraint ResultType = BroadcastType>
    std::optional<Typing::Contradiction> broadcastAndUnify(
        ValueRange exprs,
        ResultType &result,
        const llvm::Twine &what = {}) const
    {
        // Broadcast the operands, which will result in a BroadcastType.
        SmallVector<Type> broadcasted;
        if (auto contra = broadcast(exprs, broadcasted)) return contra;

        // Unify the broadcasted types to the desired BroadcastType subtype.
        return unify(broadcasted, result, what).explain(exprs);
    }

    InFlightDiagnostic emitError() const
    {
        return mlir::emitError(getParent().getLoc());
    }

private:
    std::optional<Typing::Contradiction>
    unifyImpl(SmallVectorImpl<Type> &types, Type &result) const;

    Typing::MLIRTypeChecker &m_impl;
    mlir::TypeCheckOpInterface m_parent;
};

} // namespace mlir::ekl
