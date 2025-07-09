/// Implements the TypeCheckingAdaptor.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/Analysis/TypeCheckingAdaptor.h"

#include <llvm/ADT/SmallVector.h>
#include <mlir/Typing/Contradiction.h>
#include <numeric>
#include <optional>

using namespace mlir;
using namespace mlir::ekl;

//===----------------------------------------------------------------------===//
// TypeCheckingAdaptor implementation
//===----------------------------------------------------------------------===//

std::optional<Typing::Contradiction>
TypeCheckingAdaptor::unify(ArrayRef<Type> types, Type &result) const
{
    auto temp = llvm::to_vector(types);
    return unifyImpl(temp, result);
}

std::optional<Typing::Contradiction>
TypeCheckingAdaptor::unify(ValueRange exprs, Type &result) const
{
    auto types = llvm::to_vector(exprs.getTypes());
    // auto types = getTypes(exprs);
    // FIXME: reattach note
    return unifyImpl(types, result)
        //.explain(exprs)
        ;
}

std::optional<Typing::Contradiction> TypeCheckingAdaptor::broadcast(
    ArrayRef<Type> types,
    SmallVectorImpl<extent_t> &extents) const
{
    switch (ekl::broadcast(types, extents)) {
    case BroadcastResult::Scalar:
    case BroadcastResult::Array:  return std::nullopt;
    case BroadcastResult::Unbounded:
        // return Typing::Contradiction::indeterminate();
        return std::nullopt;
    case BroadcastResult::Failure:
        /*FIXME:
        auto diag = emitError() << "can't broadcast ";
        llvm::interleaveComma(types, diag);
        diag << " together";
        return diag;
        */
        return m_impl.fatal(Typing::Source());
    }
}

std::optional<Typing::Contradiction> TypeCheckingAdaptor::broadcast(
    ValueRange exprs,
    SmallVectorImpl<extent_t> &extents) const
{
    auto types = llvm::to_vector(exprs.getTypes());
    // const auto types = getTypes(exprs);
    // FIXME: reattach note
    return broadcast(types, extents)
        //.explain(exprs)
        ;
}

std::optional<Typing::Contradiction> TypeCheckingAdaptor::broadcast(
    Type type,
    ExtentRange extents,
    ArrayType &result) const
{
    const auto maybe = ekl::broadcast(type, extents);
    if (failed(maybe)) {
        /*FIXME:
        auto diag = emitError() << "can't broadcast " << type << " to [";
        llvm::interleaveComma(extents, diag);
        diag << "]";
        return diag;
        */
        return m_impl.fatal(Typing::Source());
    }
    result = *maybe;
    return result ? std::nullopt : std::optional(Typing::Contradiction());
}

std::optional<Typing::Contradiction>
TypeCheckingAdaptor::broadcast(MutableArrayRef<Type> types) const
{
    switch (ekl::broadcast(types)) {
    case BroadcastResult::Scalar:
    case BroadcastResult::Array:     return std::nullopt;
    case BroadcastResult::Unbounded: return Typing::Contradiction();
    case BroadcastResult::Failure:
        /*FIXME
        auto diag = emitError() << "can't broadcast ";
        llvm::interleaveComma(types, diag);
        diag << " together";
        return diag;
        */
        return m_impl.fatal(Typing::Source());
    }
}

std::optional<Typing::Contradiction> TypeCheckingAdaptor::broadcast(
    ValueRange exprs,
    SmallVectorImpl<Type> &result) const
{

    auto types = llvm::to_vector(exprs.getTypes());
    // result = getTypes(exprs);
    //  FIXME: reattach note
    return broadcast(result)
        //.explain(exprs)
        ;
}

std::optional<Typing::Contradiction>
TypeCheckingAdaptor::unifyImpl(SmallVectorImpl<Type> &types, Type &result) const
{
    const auto unified = ekl::unify(types);
    if (succeeded(unified)) {
        result = *unified;
        return *unified ? std::nullopt : std::optional(Typing::Contradiction());
    }
    /*FIXME:
    auto diag = emitError() << "can't unify ";
    llvm::interleaveComma(types, diag);
    return diag;
    */
    return m_impl.fatal(Typing::Source());
}
