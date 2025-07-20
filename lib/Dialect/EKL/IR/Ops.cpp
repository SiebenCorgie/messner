/// Implementation of the EKL dialect ops.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/Ops.h"

#include "messner/Dialect/EKL/Analysis/Extent.h"
#include "messner/Dialect/EKL/Analysis/Shape.h"
#include "messner/Dialect/EKL/IR/Dialect.h"
#include "messner/Dialect/EKL/IR/TypeSystem.h"
#include "messner/Dialect/EKL/IR/Types.h"
#include "messner/Support/concepts.h"

#include <cstddef>
#include <cstdio>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/ValueRange.h>
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

/// Tries to broadcast the op's operand shape to a common shape that is also
/// compatible to the op's result.
std::optional<Typing::Contradiction> broadcastAndPromote(
    Operation *op,
    Typing::AbstractTypeChecker &typeChecker,
    Type &result)
{

    auto opTypes = op->getOperandTypes();
    auto shapes =
        llvm::to_vector(llvm::map_range(op->getOperands(), [&](auto op) {
            BroadcastType ty =
                llvm::dyn_cast_if_present<BroadcastType>(op.getType());
            assert(ty); // NOTE: assume that worked
            return ty.getShape();
        }));

    // FIXME(tendsin): Because I can't figure out a fold on the iterator,
    // above, init with first result then broadcast all shapes. Whenever
    // we encounter an _error_ return.
    Shape broadcasted = llvm::to_vector(shapes.front());
    for (auto next_shape : shapes) {
        auto ir = broadcast(next_shape, broadcasted);
        if (failed(ir)) {
            auto f = typeChecker.fatal(op->getLoc());
            f << "Invalid type shape combination: (";
            for (auto t : opTypes) f << t << " ";
            f << ") -> " << op->getResult(0).getType();
            return f;
        }
        broadcasted = ir.value();
    }

    // At this point shapes match, continue by trying to promote everything
    // i.e. tho old _unify_ stage

    auto optys = llvm::to_vector(op->getOperandTypes());
    assert(optys.size() == 2);

    // NOTE(tendsin): For some reason this triggers the default-type-constructor
    // assert in reduce-pairwise, but I can't find out why :/.
    // TODO: Reduce pairwise till we reach one common, unified type.
    //  typeChecker.getTypeSystem(op).promote(optys);
    auto unified = typeChecker.getTypeSystem(op).promote(optys[0], optys[1]);
    if (!unified) {
        auto f = typeChecker.fatal(op->getLoc());
        f << "Could not promote to unified type";
        return f;
    }
    result = unified;
    return std::nullopt;
}

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
// SubscriptOp implementation
//===----------------------------------------------------------------------===//

auto SubscriptOp::fold(FoldAdaptor) -> OpFoldResult
{
    // TODO: Implement.
    return {};
}

[[nodiscard]] static BlockArgument getInferrableIndex(Value value)
{
    // Must be a block argument.
    const auto argument = llvm::dyn_cast<BlockArgument>(value);
    if (!argument) return {};

    // Must be from the map region of an AssocOp.
    const auto owner = argument.getOwner()->getParentOp();
    if (!llvm::isa_and_present<ekl::MapOp>(*owner)) return {};
    return argument;
}

static FailureOr<ekl::IndexType> meetIndexBound(
    Typing::AbstractTypeChecker &typeChecker,
    BlockArgument index,
    Extent bound)
{
    // Update the bound on the index value, which will fail if there is already
    // a different bound.
    // FIXME(tendsin): fromEnd=false correct?
    const auto type = ekl::IndexType::get(index.getContext(), bound, false);
    auto mres       = typeChecker.meet(index, type);
    if (auto contra = mres.toContra()) return failure();

    // This invalidates the owner as well.
    typeChecker.invalidate(index.getParentRegion()->getParentOp());
    return type;
}

static std::optional<Typing::Contradiction> typeCheckSubscripts(
    Typing::AbstractTypeChecker &tc,
    ValueRange subscripts,
    SmallVectorImpl<Type> &bounds)
{
    // Check the known types of all subscript operands.
    auto unbounded = false;
    std::optional<Value> ellipsis;

    for (auto subscript : subscripts) {
        auto bound = tc.get(subscript);
        auto sTy   = subscript.getType();
        bounds.push_back(bound);

        if (!bound) {

            if (getInferrableIndex(subscript)) {
                // Will be inferred later.
                continue;
            }

            // Definitely stays unbounded.
            unbounded = true;
            continue;
        }

        if (llvm::isa<ekl::EllipsisType>(sTy)) {

            // There may only be a single ellipsis.
            if (ellipsis) {
                auto f = tc.fatal(ellipsis->getLoc());
                f << "more than one ellipsis";
                f.attachNote(subscript.getLoc()) << "found here";
                f.attachNote(ellipsis->getLoc())
                    << "previous ellipsis was here";
                return f;
            }
            ellipsis = subscript;

            continue;
        }

        // TODO(tendsin): Check that this is right, I'm assuming
        //                ExtentType == AxisType && IdentityType == SliceType
        if (llvm::isa<AxisType, SliceType>(sTy)) continue;
        // TODO(tendsin): lost the getScalarType(bound), not sure if this is
        // working as expected now...
        if (llvm::isa_and_present<ekl::IndexType>(sTy)) continue;

        // Type is not a valid indexer.
        auto f = tc.fatal(subscript.getLoc());
        f << "expected indexer, but got " << bound;
        f.attachNote(subscript.getLoc()) << "for this subscript";
        return f;
    }

    return unbounded ? std::make_optional(Typing::Contradiction())
                     : std::nullopt;
}

template<messner::type_constraint ResultType>
std::optional<Typing::Contradiction> require(
    Typing::AbstractTypeChecker &typeChecker,
    Value owner,
    Type type,
    ResultType &result)
{

    if (!result) {
        result = ResultType{};
        return Typing::Contradiction();
    }

    // Success
    if ((result = llvm::dyn_cast<ResultType>(type))) return std::nullopt;

    // Failed to cast
    auto f = typeChecker.fatal(owner.getLoc());
    f << "expected " << result;
    return f;
}

// NOTE(tendsin): cheap copy for now...
/// Gets the underlying scalar type of @p type , if any.
///
/// If @p type is a scalar matching @p ResultType , returns it. If @p type is a
/// ContiguousType over some @p ResultType , returns that type. Otherwise,
/// returns @c nullptr .
///
/// @tparam ResultType  Additional constraint on the scalar type.
///
/// @param              type    The type.
///
/// @retval nullptr     @p type is neither a scalar nor an aggregate.
/// @retval ResultType  The scalar type of @p type .
template<messner::type_constraint ResultType = ScalarType>
[[nodiscard]] inline ResultType
getScalarType(messner::type_constraint auto type)
{
    if (!type) return nullptr;
    if (const auto resultTy = llvm::dyn_cast<ResultType>(type)) return resultTy;
    if (const auto contiguousTy = llvm::dyn_cast<ContiguousType>(type))
        return llvm::dyn_cast<ResultType>(contiguousTy.getScalarType());
    return nullptr;
}

// NOTE(tendsin): cheap copy for now...
/// Gets the aggregate extents of @p type , if any.
///
/// If @p type is a ContiguousType, returns its extents. If @p type is a
/// ScalarType, returns the empty ExtentRange. Otherwise, fails.
///
/// @param              type    The type.
///
/// @retval failure     @p type is not contiguous or scalar.
/// @retval ExtentRange The extents of @p type .
inline FailureOr<ShapeRef> getExtents(messner::type_constraint auto type)
{
    if (llvm::isa_and_present<ScalarType>(type)) return ShapeRef{};
    if (const auto contiguousTy =
            llvm::dyn_cast_if_present<ContiguousType>(type))
        return contiguousTy.getShape();

    return failure();
}

auto SubscriptOp::typeCheck(Typing::AbstractTypeChecker &typeChecker)
    -> std::optional<Typing::Contradiction>
{
    // Try to unwrap the ArrayBound as an ArrayType.
    auto inTy = typeChecker.get(getArray());
    ArrayType arrayTy;
    if (auto contra = require(typeChecker, getResult(), inTy, arrayTy))
        return contra;

    // Type check subscripts
    llvm::SmallVector<Type> subscriptTys;

    if (auto contra =
            typeCheckSubscripts(typeChecker, getSubscripts(), subscriptTys)) {
        return contra;
    }

    // FIXME(tendsin): Again, I'm assuming ExtentType == AxisType
    //  Infer the result extents
    auto sourceDim = 0U;
    llvm::SmallVector<Extent> extents;
    for (auto [idx, value] : llvm::enumerate(getSubscripts())) {
        auto bound = subscriptTys[idx];
        if (llvm::isa_and_present<AxisType>(bound)) {
            // Insert a new unit dimension.
            extents.push_back(Extent(1UL));
            continue;
        } else if (llvm::isa_and_present<EllipsisType>(bound)) {
            // Count the number of remaining subscripts that will bind to a
            // source dimension.
            const auto remaining = static_cast<size_t>(llvm::count_if(
                ArrayRef<Type>(subscriptTys).drop_front(idx + 1),
                [](Type type) { return !type || !llvm::isa<AxisType>(type); }));
            // Insert the identity indexer until enough dimensions are bound.
            while (remaining < (arrayTy.getNumExtents() - sourceDim))
                extents.push_back(arrayTy.getExtent(sourceDim++));
            continue;
        }

        // For all other kinds of subscripts, bind 1 source dimension.
        if (sourceDim == arrayTy.getNumExtents()) {
            auto f = typeChecker.fatal(getLoc());
            f << "exceeded number of array extents (" << arrayTy.getNumExtents()
              << ")";
            f.attachNote(value.getLoc()) << "with this subscript";
            return f;
        }

        if (!bound) {
            const auto index = getInferrableIndex(value);
            assert(index);

            // The subscript type checker let this through because it can be
            // inferred from the array extents here.
            auto lower_extent = (arrayTy.getExtent(sourceDim) - Extent(1UL));
            // make sure it exists
            assert(lower_extent && lower_extent->isBounded());
            Extent meetExt = lower_extent.value();

            const auto meet = meetIndexBound(typeChecker, index, meetExt);
            if (failed(meet)) return typeChecker.fatal(getLoc());
            bound = *meet;
            assert(bound);
            // FIXME(tendsin): assuming IdentityType == SliceType
            // I _think_ thats not correct tho.
        } else if (llvm::isa<SliceType>(bound)) {
            // Map this dimension using the identity.
            extents.push_back(arrayTy.getExtent(sourceDim++));
            continue;
        }

        const IndexType indexTy =
            llvm::cast<ekl::IndexType>(getScalarType(bound));

        // Handle statically known index bounds.
        if (indexTy.isBounded()
            && indexTy.getBound() >= arrayTy.getExtent(sourceDim)) {
            auto f = typeChecker.fatal(getLoc());
            f << "index out of bounds (" << indexTy.getBound().getValue()
              << " >= " << arrayTy.getExtent(sourceDim).getValue() << ")";
            f.attachNote(value.getLoc()) << "for this subscript";
            return f;
        }

        // Insert the indexer's extents here, and skip this dimension in the
        // source.
        ++sourceDim;
        concat(extents, getExtents(bound).value_or(ShapeRef{}));
    }

    // For partial subscripting, append all the remaining dimensions.
    concat(extents, arrayTy.getShape().drop_front(sourceDim));

    // The result is an array with the inferred extents, decaying to a scalar
    // if the extents are empty.
    Type resultTy = arrayTy.getScalarType();
    if (!extents.empty()) resultTy = ArrayType::get(resultTy, extents);

    auto mres = typeChecker.meet(getResult(), resultTy);
    if (auto contra = mres.toContra()) return contra;
    // success
    return std::nullopt;
}

//===----------------------------------------------------------------------===//
// StackOp implementation
//===----------------------------------------------------------------------===//

auto StackOp::fold(FoldAdaptor) -> OpFoldResult
{
    // TODO: Implement.
    return {};
}

auto StackOp::typeCheck(Typing::AbstractTypeChecker &typeChecker)
    -> std::optional<Typing::Contradiction>
{
    // must BroadcastAndUnify
    BroadcastType unfiedTy;
    if (auto contra =
            broadcastAndPromote(getOperation(), typeChecker, unfiedTy))
        return contra;

    // The result extents are determined by the unified extents, prepended by
    // the number of stacked atoms.
    auto resultExtents =
        concat(Extent(getOperands().size()), unfiedTy.getShape());

    auto mres =
        typeChecker.meet(getResult(), unfiedTy.cloneWith(resultExtents));
    if (auto maybeContra = mres.toContra()) return maybeContra;

    return std::nullopt;
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

//===----------------------------------------------------------------------===//
// Compare operator implementation
//===----------------------------------------------------------------------===//

auto CompareOp::fold(FoldAdaptor) -> OpFoldResult
{
    // TODO: Implement.
    return {};
}

std::optional<Typing::Contradiction>
CompareOp::typeCheck(::mlir::Typing::AbstractTypeChecker &typeChecker)
{
    BroadcastType unifiedTy;
    if (auto contra =
            broadcastAndPromote(getOperation(), typeChecker, unifiedTy)) {
        return contra;
    }

    if (!llvm::isa<NumberType>(unifiedTy.getScalarType())) {
        switch (getKind()) {
        case RelationKind::Equivalent:
        case RelationKind::Antivalent:
            if (llvm::isa<BoolType>(unifiedTy.getScalarType())) {
                // Equivalence/Antivalence of booleans is also well-defined.
                break;
            }
            [[fallthrough]];

        default:
            auto f = typeChecker.fatal(getLoc());
            f << "can't relate values of type " << unifiedTy;
            return f;
        }
    }

    // We did meet the requirements (i.e. broadcast+met input types with a well
    // defined numeric unified type.) therefore just set the result to bool with
    // the same shape.
    auto result = typeChecker.meet(
        getResult(),
        unifiedTy.cloneWith(BoolType::get(getContext())));
    if (auto contra = result.toContra()) {
        contra->attachNote(getLoc());
        return contra;
    } else
        return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Logical operator implementation
//===----------------------------------------------------------------------===//

static std::optional<Typing::Contradiction>
typeCheckLogicalOp(Operation *op, Typing::AbstractTypeChecker &typeChecker)
{

    LogicType unifiedTy;
    if (auto contra = broadcastAndPromote(op, typeChecker, unifiedTy))
        return contra;

    // The result type is the unified type, decayed to a scalar.
    if (auto result = typeChecker.meet(op->getResult(0), unifiedTy)) {
        if (auto contra = result.toContra()) {
            // failed to meet unified
            return contra;
        } else {
            return std::nullopt;
        }
    } else {
        auto f = typeChecker.fatal(op->getLoc());
        f << "Could not deduce type";
        return f;
    }
}

auto LogicalNotOp::fold(FoldAdaptor) -> OpFoldResult
{
    // TODO: Implement.
    return {};
}

std::optional<Typing::Contradiction>
LogicalNotOp::typeCheck(::mlir::Typing::AbstractTypeChecker &typeChecker)
{
    // NOTE(tendsin): I think this should be trivial? I.e. a single operand that
    // either meets the result's
    //                requirement, or not
    assert(getOperation()->getOperands().size() == 1);
    auto result = typeChecker.meet(getResult(), typeChecker.get(getOperand()));
    if (auto contra = result.toContra()) {
        contra->attachNote(getLoc());
        return contra;
    } else
        return std::nullopt;
}

auto LogicalOrOp::fold(FoldAdaptor) -> OpFoldResult
{
    // TODO: Implement.
    return {};
}

std::optional<Typing::Contradiction>
LogicalOrOp::typeCheck(::mlir::Typing::AbstractTypeChecker &typeChecker)
{
    return typeCheckLogicalOp(getOperation(), typeChecker);
}

auto LogicalAndOp::fold(FoldAdaptor) -> OpFoldResult
{
    // TODO: Implement.
    return {};
}

std::optional<Typing::Contradiction>
LogicalAndOp::typeCheck(::mlir::Typing::AbstractTypeChecker &typeChecker)
{
    return typeCheckLogicalOp(getOperation(), typeChecker);
}

//===----------------------------------------------------------------------===//
// Arithmetic operator implementation
//===----------------------------------------------------------------------===//

static std::optional<Typing::Contradiction> typeCheckArithmeticOp(
    Operation *op,
    ::mlir::Typing::AbstractTypeChecker &typeChecker,
    function_ref<Extent(ArrayRef<Extent>)> combineIndexBounds)
{

    ArithmeticType unifiedTy;
    if (auto contra = broadcastAndPromote(op, typeChecker, unifiedTy))
        return contra;

    if (unifiedTy == Type{}) {
        auto f = typeChecker.fatal(op->getLoc());
        f << "Could not resolve to common result type";
        return f;
    }

    // Arithmetic operations need to properly update the upper bounds on the
    // types of index values they produce.
    if (llvm::isa<ekl::IndexType>(unifiedTy)) {
        // Flatten each shape into an extent, then use the λ to come up with a
        // unified bound. Finally obtain the IndexType with such an extent and
        // tell the type checker.
        auto shape_extents =
            llvm::to_vector(llvm::map_range(op->getOperands(), [&](auto op) {
                BroadcastType ty =
                    llvm::dyn_cast_if_present<BroadcastType>(op.getType());
                assert(ty); // NOTE: assume that worked
                auto shape     = ty.getShape();
                // Flatten the shape into one extent
                auto flattened = flatten(shape);
                assert(llvm::succeeded(flattened)); // Assume this works atm.
                return flattened.value();
            }));

        auto combined_bound = combineIndexBounds(shape_extents);
        // FIXME(tendsin): isFromEnd not considered atm.
        auto unified_index_ty =
            ekl::IndexType::get(unifiedTy.getContext(), combined_bound, false);
        auto mresult = typeChecker.meet(op->getResult(0), unified_index_ty);
    }

    // Finally meet the unified type.
    if (auto result = typeChecker.meet(op->getResult(0), unifiedTy)) {
        if (auto contra = result.toContra()) {
            // failed to meet unified
            return contra;
        } else {
            return std::nullopt;
        }
    } else {
        auto f = typeChecker.fatal(op->getLoc());
        f << "Could not deduce type";
        return f;
    }
}

auto AddOp::fold(FoldAdaptor) -> OpFoldResult
{
    // TODO: Implement.
    return {};
}

std::optional<Typing::Contradiction>
AddOp::typeCheck(::mlir::Typing::AbstractTypeChecker &typeChecker)
{
    return typeCheckArithmeticOp(
        getOperation(),
        typeChecker,
        [](ArrayRef<Extent> bounds) -> Extent {
            return (bounds[0] + bounds[1]).value_or(Extent(unbounded_t{}));
        });
}

auto SubtractOp::fold(FoldAdaptor) -> OpFoldResult
{
    // TODO: Implement.
    return {};
}

std::optional<Typing::Contradiction>
SubtractOp::typeCheck(::mlir::Typing::AbstractTypeChecker &typeChecker)
{
    return typeCheckArithmeticOp(
        getOperation(),
        typeChecker,
        [](ArrayRef<Extent> bounds) -> Extent { return bounds[0]; });
}

auto MultiplyOp::fold(FoldAdaptor) -> OpFoldResult
{
    // TODO: Implement.
    return {};
}

std::optional<Typing::Contradiction>
MultiplyOp::typeCheck(::mlir::Typing::AbstractTypeChecker &typeChecker)
{
    return typeCheckArithmeticOp(
        getOperation(),
        typeChecker,
        [](ArrayRef<Extent> bounds) -> Extent {
            if (!bounds[0].isBounded() || !bounds[1].isBounded())
                return Extent(unbounded_t{});
            return (bounds[0] * bounds[1].getValue())
                .value_or(Extent(unbounded_t{}));
        });
}

auto DivideOp::fold(FoldAdaptor) -> OpFoldResult
{
    // TODO: Implement.
    return {};
}

std::optional<Typing::Contradiction>
DivideOp::typeCheck(::mlir::Typing::AbstractTypeChecker &typeChecker)
{
    return typeCheckArithmeticOp(
        getOperation(),
        typeChecker,
        [](ArrayRef<Extent> bounds) -> Extent { return bounds[0]; });
}

auto RemainderOp::fold(FoldAdaptor) -> OpFoldResult
{
    // TODO: Implement.
    return {};
}

std::optional<Typing::Contradiction>
RemainderOp::typeCheck(::mlir::Typing::AbstractTypeChecker &typeChecker)
{
    return typeCheckArithmeticOp(
        getOperation(),
        typeChecker,
        [](ArrayRef<Extent> bounds) -> Extent {
            if (!bounds[1].isBounded()) return Extent(unbounded_t{});
            return (bounds[1] - 1).value_or(Extent(unbounded_t{}));
        });
}

auto PowerOp::fold(FoldAdaptor) -> OpFoldResult
{
    // TODO: Implement.
    return {};
}

std::optional<Typing::Contradiction>
PowerOp::typeCheck(::mlir::Typing::AbstractTypeChecker &typeChecker)
{
    return typeCheckArithmeticOp(
        getOperation(),
        typeChecker,
        [](ArrayRef<Extent> bounds) -> Extent {
            return Extent(unbounded_t{});
        });
}

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
    return typeCheckArithmeticOp(
        getOperation(),
        typeChecker,
        [](ArrayRef<Extent> bounds) -> Extent {
            return std::min(bounds[0], bounds[1]);
        });
}
//===----------------------------------------------------------------------===//
// MaxOp implementation
//===----------------------------------------------------------------------===//

auto MaxOp::fold(FoldAdaptor) -> OpFoldResult
{
    // TODO: Implement.
    return {};
}

std::optional<Typing::Contradiction>
MaxOp::typeCheck(::mlir::Typing::AbstractTypeChecker &typeChecker)
{
    return typeCheckArithmeticOp(
        getOperation(),
        typeChecker,
        [](ArrayRef<Extent> bounds) -> Extent {
            return std::max(bounds[0], bounds[1]);
        });
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
