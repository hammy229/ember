#include "ember/tensor.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace {

using ember::Arena;
using ember::DType;
using ember::Shape;
using ember::Tensor;
using ember::TensorView;

// Writing through a TensorView must not compile. This is the promise that protects weights mapped
// from a read-only file.
static_assert(std::is_same_v<decltype(std::declval<TensorView>().data<float>()), const float*>);
static_assert(std::is_same_v<decltype(std::declval<TensorView>().at<float>(0)), const float&>);
static_assert(std::is_same_v<decltype(std::declval<Tensor>().data<float>()), float*>);
static_assert(std::is_same_v<decltype(std::declval<Tensor>().at<float>(0)), float&>);
static_assert(std::is_convertible_v<Tensor, TensorView>, "reading a Tensor must just work");
static_assert(!std::is_convertible_v<TensorView, Tensor>, "a view must not become writable");
// Passing tensors around must stay as cheap as copying a few words.
static_assert(std::is_trivially_copyable_v<TensorView>);
static_assert(std::is_trivially_copyable_v<Tensor>);

// A 2 by 3 tensor counting 0 to 5, which makes wrong offsets easy to spot.
Tensor Counting2x3(Arena& arena) {
  const std::vector<float> values{0, 1, 2, 3, 4, 5};
  return Tensor::FromValues<float>(arena, *Shape::Make({2, 3}), values).value();
}

}  // namespace

TEST(Tensor, DefaultIsEmpty) {
  const Tensor tensor;
  EXPECT_EQ(tensor.element_count(), 0U);
  EXPECT_EQ(tensor.byte_size(), 0U);
  EXPECT_EQ(tensor.dtype(), DType::kF32);
  EXPECT_EQ(tensor.ToString(), "f32[0]");
}

TEST(Tensor, UninitializedHasTheRightShape) {
  Arena arena;
  const auto tensor = Tensor::Uninitialized(arena, DType::kI32, *Shape::Make({2, 5}));
  ASSERT_TRUE(tensor.ok());
  EXPECT_EQ(tensor->dtype(), DType::kI32);
  EXPECT_EQ(tensor->element_count(), 10U);
  EXPECT_EQ(tensor->byte_size(), 40U);
  EXPECT_EQ(tensor->rank(), 2U);
  EXPECT_EQ(tensor->dim(1), 5U);
}

TEST(Tensor, MemoryIsCacheLineAligned) {
  Arena arena;
  for (int i = 0; i < 5; ++i) {
    const auto tensor = Tensor::Uninitialized(arena, DType::kI8, *Shape::Make({3}));
    ASSERT_TRUE(tensor.ok());
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(tensor->raw_data()) % 64, 0U);
  }
}

TEST(Tensor, ZerosIsAllZero) {
  Arena arena;
  // Dirty the memory first, so a zero below cannot be luck.
  const Arena::Mark mark = arena.mark();
  const auto dirty = arena.Allocate(1024);
  ASSERT_TRUE(dirty.ok());
  std::fill_n(*dirty, 1024, std::byte{0xFF});
  arena.Rewind(mark);

  const auto tensor = Tensor::Zeros(arena, DType::kF32, *Shape::Make({16}));
  ASSERT_TRUE(tensor.ok());
  for (std::size_t i = 0; i < 16; ++i) EXPECT_EQ(tensor->at<float>(i), 0.0F);
}

TEST(Tensor, StridesAreRowByRow) {
  Arena arena;
  const auto tensor = Tensor::Uninitialized(arena, DType::kF32, *Shape::Make({2, 3, 4}));
  ASSERT_TRUE(tensor.ok());
  EXPECT_EQ(tensor->stride(0), 12U);
  EXPECT_EQ(tensor->stride(1), 4U);
  EXPECT_EQ(tensor->stride(2), 1U);
}

TEST(Tensor, AtFollowsTheStrides) {
  Arena arena;
  const Tensor tensor = Counting2x3(arena);
  EXPECT_EQ(tensor.at<float>(0, 0), 0.0F);
  EXPECT_EQ(tensor.at<float>(0, 2), 2.0F);
  EXPECT_EQ(tensor.at<float>(1, 0), 3.0F);
  EXPECT_EQ(tensor.at<float>(1, 2), 5.0F);
}

TEST(Tensor, AtCanWrite) {
  Arena arena;
  const Tensor tensor = Counting2x3(arena);
  tensor.at<float>(1, 1) = 42.0F;
  EXPECT_EQ(tensor.data<float>()[4], 42.0F);
}

TEST(Tensor, ScalarHasOneElement) {
  Arena arena;
  const std::array<float, 1> value{7.0F};
  const auto tensor = Tensor::FromValues<float>(arena, Shape{}, value);
  ASSERT_TRUE(tensor.ok());
  EXPECT_EQ(tensor->rank(), 0U);
  EXPECT_EQ(tensor->element_count(), 1U);
  EXPECT_EQ(tensor->at<float>(), 7.0F);
}

TEST(Tensor, EmptyTensorWorksAndGetsAPointer) {
  Arena arena;
  const auto tensor = Tensor::Zeros(arena, DType::kF32, *Shape::Make({0, 8}));
  ASSERT_TRUE(tensor.ok());
  EXPECT_EQ(tensor->byte_size(), 0U);
  EXPECT_EQ(tensor->stride(0), 8U);
}

TEST(Tensor, FromValuesRejectsTheWrongCount) {
  Arena arena;
  const std::vector<float> values{1, 2, 3};
  const auto tensor = Tensor::FromValues<float>(arena, *Shape::Make({2, 2}), values);
  ASSERT_FALSE(tensor.ok());
  EXPECT_EQ(tensor.status().code(), ember::StatusCode::kInvalidArgument);
}

TEST(Tensor, ReportsWhenTheArenaIsFull) {
  Arena::Options options;
  options.max_bytes = 1024;
  Arena arena(options);
  const auto tensor = Tensor::Zeros(arena, DType::kF32, *Shape::Make({1024}));
  ASSERT_FALSE(tensor.ok());
  EXPECT_EQ(tensor.status().code(), ember::StatusCode::kOutOfMemory);
}

TEST(Tensor, TheLargestShapeFailsCleanly) {
  // element_count * 8 still fits in a size_t, so the arena gets an honest request it can refuse.
  Arena arena;
  const auto tensor =
      Tensor::Uninitialized(arena, DType::kF32, *Shape::Make({Shape::kMaxElements}));
  ASSERT_FALSE(tensor.ok());
  EXPECT_EQ(tensor.status().code(), ember::StatusCode::kOutOfMemory);
}

TEST(Tensor, WrapsOutsideMemoryWithoutCopying) {
  std::array<float, 4> outside{1, 2, 3, 4};
  const auto tensor = Tensor::Wrap(outside.data(), DType::kF32, *Shape::Make({2, 2}));
  ASSERT_TRUE(tensor.ok());
  tensor->at<float>(1, 1) = 9.0F;
  EXPECT_EQ(outside[3], 9.0F);
}

TEST(TensorView, WrapsReadOnlyMemory) {
  static constexpr std::array<std::int32_t, 3> kTokens{10, 20, 30};
  const auto view = TensorView::Wrap(kTokens.data(), DType::kI32, *Shape::Make({3}));
  ASSERT_TRUE(view.ok());
  EXPECT_EQ(view->at<std::int32_t>(2), 30);
}

TEST(TensorView, WrapRejectsNullWhenThereAreElements) {
  const auto view = TensorView::Wrap(nullptr, DType::kF32, *Shape::Make({2}));
  ASSERT_FALSE(view.ok());
  EXPECT_EQ(view.status().code(), ember::StatusCode::kInvalidArgument);
}

TEST(TensorView, WrapAllowsNullWhenThereAreNoElements) {
  EXPECT_TRUE(TensorView::Wrap(nullptr, DType::kF32, *Shape::Make({0})).ok());
}

TEST(TensorView, WrapRejectsMisalignedPointers) {
  alignas(4) std::array<std::byte, 16> bytes{};
  const auto misaligned = TensorView::Wrap(bytes.data() + 1, DType::kF32, *Shape::Make({2}));
  ASSERT_FALSE(misaligned.ok());
  EXPECT_EQ(misaligned.status().code(), ember::StatusCode::kInvalidArgument);
  // The same pointer is fine for one byte elements.
  EXPECT_TRUE(TensorView::Wrap(bytes.data() + 1, DType::kI8, *Shape::Make({2})).ok());
}

TEST(TensorView, WrapRejectsAnUnknownElementType) {
  // This used to divide by zero, and the element type can come from a corrupt model file.
  alignas(8) std::array<std::byte, 16> bytes{};
  const auto unknown = static_cast<DType>(9);
  const auto view = TensorView::Wrap(bytes.data(), unknown, *Shape::Make({2}));
  ASSERT_FALSE(view.ok());
  EXPECT_EQ(view.status().code(), ember::StatusCode::kInvalidFormat);

  Arena arena;
  EXPECT_FALSE(Tensor::Zeros(arena, unknown, *Shape::Make({2})).ok());
}

TEST(TensorView, WrapWithASizeAcceptsAnExactFit) {
  alignas(8) std::array<std::byte, 24> bytes{};
  const auto view =
      TensorView::Wrap(std::span<const std::byte>(bytes), DType::kF32, *Shape::Make({2, 3}));
  ASSERT_TRUE(view.ok());
  EXPECT_EQ(view->byte_size(), 24U);
}

TEST(TensorView, WrapWithASizeRejectsAShapeLargerThanTheBytes) {
  // A file that claims more numbers than it holds. Without this check the claim would only show up
  // later, as a read past the end of the mapping.
  alignas(8) std::array<std::byte, 24> bytes{};
  const auto view =
      TensorView::Wrap(std::span<const std::byte>(bytes), DType::kF32, *Shape::Make({2, 4}));
  ASSERT_FALSE(view.ok());
  EXPECT_EQ(view.status().code(), ember::StatusCode::kInvalidFormat);
}

TEST(TensorView, WrapWithASizeRejectsTheLargestShape) {
  alignas(8) std::array<std::byte, 8> bytes{};
  EXPECT_FALSE(TensorView::Wrap(std::span<const std::byte>(bytes), DType::kF32,
                                *Shape::Make({Shape::kMaxElements}))
                   .ok());
}

TEST(TensorView, ATensorConvertsToAView) {
  Arena arena;
  const Tensor tensor = Counting2x3(arena);
  const TensorView view = tensor;
  EXPECT_EQ(view.at<float>(1, 2), 5.0F);
  EXPECT_EQ(view.shape(), tensor.shape());
}

TEST(TensorView, CopiesLookAtTheSameNumbers) {
  Arena arena;
  const Tensor tensor = Counting2x3(arena);
  const Tensor copy = tensor;  // NOLINT: the copy is what is being tested.
  copy.at<float>(0, 0) = 100.0F;
  EXPECT_EQ(tensor.at<float>(0, 0), 100.0F);
}

TEST(TensorView, ToStringShowsTypeAndShape) {
  Arena arena;
  EXPECT_EQ(Counting2x3(arena).ToString(), "f32[2, 3]");
}

TEST(TensorViews, AFreshTensorIsContiguous) {
  Arena arena;
  EXPECT_TRUE(Counting2x3(arena).is_contiguous());
}

TEST(TensorViews, SliceARowStaysContiguous) {
  Arena arena;
  const auto row = Counting2x3(arena).Slice(0, 1, 1);
  ASSERT_TRUE(row.ok());
  EXPECT_EQ(row->ToString(), "f32[1, 3]");
  EXPECT_TRUE(row->is_contiguous());
  EXPECT_EQ(row->at<float>(0, 0), 3.0F);
  EXPECT_EQ(row->at<float>(0, 2), 5.0F);
}

TEST(TensorViews, SliceAColumnIsNotContiguous) {
  Arena arena;
  const auto column = Counting2x3(arena).Slice(1, 2, 1);
  ASSERT_TRUE(column.ok());
  EXPECT_EQ(column->ToString(), "f32[2, 1]");
  EXPECT_EQ(column->at<float>(0, 0), 2.0F);
  EXPECT_EQ(column->at<float>(1, 0), 5.0F);
  EXPECT_FALSE(column->is_contiguous());
}

TEST(TensorViews, SliceKeepsSharingTheNumbers) {
  Arena arena;
  const Tensor tensor = Counting2x3(arena);
  const auto column = tensor.Slice(1, 1, 2);
  ASSERT_TRUE(column.ok());
  column->at<float>(1, 1) = 99.0F;
  EXPECT_EQ(tensor.at<float>(1, 2), 99.0F);
}

TEST(TensorViews, SlicesOfSlicesAddUp) {
  Arena arena;
  const std::vector<float> values{0, 1, 2, 3, 4, 5, 6, 7};
  const Tensor tensor = *Tensor::FromValues<float>(arena, *Shape::Make({8}), values);
  const auto inner = tensor.Slice(0, 2, 5)->Slice(0, 1, 2);
  ASSERT_TRUE(inner.ok());
  EXPECT_EQ(inner->at<float>(0), 3.0F);
  EXPECT_EQ(inner->at<float>(1), 4.0F);
}

TEST(TensorViews, EmptySliceIsAllowed) {
  Arena arena;
  const auto empty = Counting2x3(arena).Slice(1, 3, 0);
  ASSERT_TRUE(empty.ok());
  EXPECT_EQ(empty->element_count(), 0U);
  EXPECT_TRUE(empty->is_contiguous());
}

TEST(TensorViews, SlicingANullEmptyTensorStaysSafe) {
  // UBSan would flag moving the null pointer here.
  const auto view = TensorView::Wrap(nullptr, DType::kF32, *Shape::Make({0, 8}));
  ASSERT_TRUE(view.ok());
  const auto slice = view->Slice(1, 2, 3);
  ASSERT_TRUE(slice.ok());
  EXPECT_EQ(slice->raw_data(), nullptr);
}

TEST(TensorViews, SliceRejectsWhatDoesNotFit) {
  Arena arena;
  const Tensor tensor = Counting2x3(arena);
  const std::size_t huge = std::numeric_limits<std::size_t>::max();
  EXPECT_FALSE(tensor.Slice(2, 0, 1).ok());     // No such dimension.
  EXPECT_FALSE(tensor.Slice(0, 3, 0).ok());     // Starts past the end.
  EXPECT_FALSE(tensor.Slice(0, 1, 2).ok());     // Runs past the end.
  EXPECT_FALSE(tensor.Slice(0, huge, 2).ok());  // start + length would wrap to a small number.
  EXPECT_FALSE(tensor.Slice(0, 1, huge).ok());
}

TEST(TensorViews, ScalarsCannotBeSliced) {
  const std::array<float, 1> value{1.0F};
  const auto view = TensorView::Wrap(value.data(), DType::kF32, Shape{});
  ASSERT_TRUE(view.ok());
  EXPECT_FALSE(view->Slice(0, 0, 1).ok());
}

TEST(TensorViews, TransposeSwapsWithoutMovingNumbers) {
  Arena arena;
  const Tensor tensor = Counting2x3(arena);
  const auto swapped = tensor.Transpose(0, 1);
  ASSERT_TRUE(swapped.ok());
  EXPECT_EQ(swapped->ToString(), "f32[3, 2]");
  EXPECT_EQ(swapped->raw_data(), tensor.raw_data());
  EXPECT_FALSE(swapped->is_contiguous());
  for (std::size_t row = 0; row < 2; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      EXPECT_EQ(swapped->at<float>(column, row), tensor.at<float>(row, column));
    }
  }
}

TEST(TensorViews, TransposingTwiceGivesTheOriginalBack) {
  Arena arena;
  const Tensor tensor = Counting2x3(arena);
  const auto back = tensor.Transpose(0, 1)->Transpose(0, 1);
  ASSERT_TRUE(back.ok());
  EXPECT_EQ(back->shape(), tensor.shape());
  EXPECT_TRUE(back->is_contiguous());
  EXPECT_EQ(back->at<float>(1, 2), 5.0F);
}

TEST(TensorViews, TransposeWorksOnFourDimensions) {
  // [batch, tokens, heads, features] to [batch, heads, tokens, features], as attention does.
  Arena arena;
  const auto tensor = Tensor::Zeros(arena, DType::kF32, *Shape::Make({1, 2, 3, 4}));
  ASSERT_TRUE(tensor.ok());
  tensor->at<float>(0, 1, 2, 3) = 8.0F;
  const auto swapped = tensor->Transpose(1, 2);
  ASSERT_TRUE(swapped.ok());
  EXPECT_EQ(swapped->at<float>(0, 2, 1, 3), 8.0F);
}

TEST(TensorViews, TransposeRejectsMissingDimensions) {
  Arena arena;
  EXPECT_FALSE(Counting2x3(arena).Transpose(0, 2).ok());
  EXPECT_FALSE(Counting2x3(arena).Transpose(2, 0).ok());
}

TEST(TensorViews, ReshapeKeepsTheOrderOfTheNumbers) {
  Arena arena;
  const Tensor tensor = Counting2x3(arena);
  const auto reshaped = tensor.Reshape(*Shape::Make({3, 2}));
  ASSERT_TRUE(reshaped.ok());
  EXPECT_EQ(reshaped->raw_data(), tensor.raw_data());
  EXPECT_EQ(reshaped->at<float>(1, 0), 2.0F);
  EXPECT_EQ(reshaped->at<float>(2, 1), 5.0F);
}

TEST(TensorViews, ReshapeCanFlattenAndAddDimensions) {
  Arena arena;
  const auto flat = Counting2x3(arena).Reshape(*Shape::Make({6}));
  ASSERT_TRUE(flat.ok());
  EXPECT_EQ(flat->at<float>(5), 5.0F);
  const auto wider = flat->Reshape(*Shape::Make({1, 6, 1}));
  ASSERT_TRUE(wider.ok());
  EXPECT_EQ(wider->at<float>(0, 5, 0), 5.0F);
}

TEST(TensorViews, ReshapeRejectsADifferentCount) {
  Arena arena;
  const auto reshaped = Counting2x3(arena).Reshape(*Shape::Make({4, 2}));
  ASSERT_FALSE(reshaped.ok());
  EXPECT_EQ(reshaped.status().code(), ember::StatusCode::kInvalidArgument);
}

TEST(TensorViews, ReshapeRefusesNumbersThatAreNotContiguous) {
  Arena arena;
  const auto swapped = Counting2x3(arena).Transpose(0, 1);
  ASSERT_TRUE(swapped.ok());
  const auto reshaped = swapped->Reshape(*Shape::Make({6}));
  ASSERT_FALSE(reshaped.ok());
  EXPECT_EQ(reshaped.status().code(), ember::StatusCode::kInvalidArgument);
}

TEST(TensorViews, AViewOfAViewCannotWrite) {
  static_assert(std::is_same_v<decltype(std::declval<TensorView>().Slice(0, 0, 1)),
                               ember::StatusOr<TensorView>>);
  static_assert(
      std::is_same_v<decltype(std::declval<Tensor>().Slice(0, 0, 1)), ember::StatusOr<Tensor>>);
}

TEST(TensorCopy, CopiesContiguousNumbers) {
  Arena arena;
  const Tensor source = Counting2x3(arena);
  const Tensor target = *Tensor::Zeros(arena, DType::kF32, source.shape());
  ASSERT_TRUE(target.CopyFrom(source).ok());
  EXPECT_EQ(target.at<float>(1, 2), 5.0F);
  EXPECT_NE(target.raw_data(), source.raw_data());
}

TEST(TensorCopy, FollowsTheStridesOfATranspose) {
  Arena arena;
  const Tensor source = Counting2x3(arena);
  const auto swapped = source.Transpose(0, 1);
  ASSERT_TRUE(swapped.ok());
  const Tensor target = *Tensor::Zeros(arena, DType::kF32, swapped->shape());
  ASSERT_TRUE(target.CopyFrom(*swapped).ok());
  EXPECT_TRUE(target.is_contiguous());
  // The numbers are now physically in column order.
  const std::array<float, 6> expected{0, 3, 1, 4, 2, 5};
  for (std::size_t i = 0; i < expected.size(); ++i) EXPECT_EQ(target.data<float>()[i], expected[i]);
}

TEST(TensorCopy, CanWriteIntoASlice) {
  Arena arena;
  const Tensor target = *Tensor::Zeros(arena, DType::kF32, *Shape::Make({2, 3}));
  const std::vector<float> column{7, 8};
  const Tensor source = *Tensor::FromValues<float>(arena, *Shape::Make({2, 1}), column);
  const auto slot = target.Slice(1, 1, 1);
  ASSERT_TRUE(slot.ok());
  ASSERT_TRUE(slot->CopyFrom(source).ok());
  EXPECT_EQ(target.at<float>(0, 0), 0.0F);
  EXPECT_EQ(target.at<float>(0, 1), 7.0F);
  EXPECT_EQ(target.at<float>(1, 1), 8.0F);
  EXPECT_EQ(target.at<float>(1, 2), 0.0F);
}

TEST(TensorCopy, HandlesScalarsAndEmptyTensors) {
  Arena arena;
  const std::array<float, 1> value{3.0F};
  const Tensor scalar = *Tensor::FromValues<float>(arena, Shape{}, value);
  const Tensor target = *Tensor::Zeros(arena, DType::kF32, Shape{});
  ASSERT_TRUE(target.CopyFrom(scalar).ok());
  EXPECT_EQ(target.at<float>(), 3.0F);

  const Tensor empty = *Tensor::Zeros(arena, DType::kF32, *Shape::Make({0, 2}));
  EXPECT_TRUE(empty.CopyFrom(empty).ok());
}

TEST(TensorCopy, RejectsADifferentShapeOrType) {
  Arena arena;
  const Tensor source = Counting2x3(arena);
  const Tensor wrong_shape = *Tensor::Zeros(arena, DType::kF32, *Shape::Make({3, 2}));
  const Tensor wrong_type = *Tensor::Zeros(arena, DType::kI32, source.shape());
  EXPECT_FALSE(wrong_shape.CopyFrom(source).ok());
  EXPECT_FALSE(wrong_type.CopyFrom(source).ok());
}

TEST(TensorCopy, RefusesToCopyOntoItself) {
  Arena arena;
  const std::vector<float> values{1, 2, 3, 4};
  const Tensor square = *Tensor::FromValues<float>(arena, *Shape::Make({2, 2}), values);
  const auto swapped = square.Transpose(0, 1);
  ASSERT_TRUE(swapped.ok());

  const ember::Status status = square.CopyFrom(*swapped);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), ember::StatusCode::kInvalidArgument);
  // The refusal happened before any number moved.
  EXPECT_EQ(square.at<float>(0, 1), 2.0F);
  EXPECT_EQ(square.at<float>(1, 0), 3.0F);
}

TEST(TensorCopy, AllowsNeighborsThatDoNotOverlap) {
  Arena arena;
  const std::vector<float> values{1, 2, 3, 4};
  const Tensor line = *Tensor::FromValues<float>(arena, *Shape::Make({4}), values);
  const auto left = line.Slice(0, 0, 2);
  const auto right = line.Slice(0, 2, 2);
  ASSERT_TRUE(left.ok() && right.ok());
  ASSERT_TRUE(left->CopyFrom(*right).ok());
  EXPECT_EQ(line.at<float>(0), 3.0F);
  EXPECT_EQ(line.at<float>(1), 4.0F);
}

TEST(TensorCopy, RefusesSlicesThatOverlapByOneElement) {
  Arena arena;
  const std::vector<float> values{1, 2, 3};
  const Tensor line = *Tensor::FromValues<float>(arena, *Shape::Make({3}), values);
  EXPECT_FALSE(line.Slice(0, 0, 2)->CopyFrom(*line.Slice(0, 1, 2)).ok());
}

TEST(TensorCopy, CopyOfMakesATransposeContiguous) {
  Arena arena;
  const Tensor source = Counting2x3(arena);
  const auto swapped = source.Transpose(0, 1);
  ASSERT_TRUE(swapped.ok());
  ASSERT_FALSE(swapped->Reshape(*Shape::Make({6})).ok());

  const auto copy = Tensor::CopyOf(arena, *swapped);
  ASSERT_TRUE(copy.ok());
  const auto flat = copy->Reshape(*Shape::Make({6}));
  ASSERT_TRUE(flat.ok());
  EXPECT_EQ(flat->at<float>(1), 3.0F);
}

TEST(TensorCopy, CopyOfIsIndependentOfTheOriginal) {
  Arena arena;
  const Tensor source = Counting2x3(arena);
  const Tensor copy = *Tensor::CopyOf(arena, source);
  source.at<float>(0, 0) = 50.0F;
  EXPECT_EQ(copy.at<float>(0, 0), 0.0F);
}

TEST(TensorFill, SetsEveryElement) {
  Arena arena;
  const Tensor tensor = Counting2x3(arena);
  tensor.Fill(2.5F);
  for (std::size_t i = 0; i < 6; ++i) EXPECT_EQ(tensor.data<float>()[i], 2.5F);
}

TEST(TensorFill, TouchesOnlyTheSlice) {
  Arena arena;
  const Tensor tensor = Counting2x3(arena);
  tensor.Slice(1, 2, 1)->Fill(-1.0F);
  const std::array<float, 6> expected{0, 1, -1, 3, 4, -1};
  for (std::size_t i = 0; i < expected.size(); ++i) EXPECT_EQ(tensor.data<float>()[i], expected[i]);
}

TEST(TensorFill, WorksOnScalarsAndEmptyTensorsAndOtherTypes) {
  Arena arena;
  const Tensor scalar = *Tensor::Zeros(arena, DType::kI32, Shape{});
  scalar.Fill(std::int32_t{9});
  EXPECT_EQ(scalar.at<std::int32_t>(), 9);

  const Tensor empty = *Tensor::Zeros(arena, DType::kI8, *Shape::Make({0}));
  empty.Fill(std::int8_t{1});
  EXPECT_EQ(empty.element_count(), 0U);
}

TEST(TensorDeathTest, FillingWithTheWrongTypeIsABug) {
  Arena arena;
  const Tensor tensor = Counting2x3(arena);
  EXPECT_DEATH(tensor.Fill(std::int32_t{1}), "wrong element type");
}

TEST(TensorDeathTest, ReadingAsTheWrongTypeIsABug) {
  Arena arena;
  const Tensor tensor = Counting2x3(arena);
  EXPECT_DEATH((void)tensor.data<std::int32_t>(), "wrong element type");
}

TEST(TensorDeathTest, AskingForAStridePastTheRankIsABug) {
  Arena arena;
  const Tensor tensor = Counting2x3(arena);
  EXPECT_DEATH((void)tensor.stride(2), "past the end of a tensor");
}

#ifndef NDEBUG
TEST(TensorDeathTest, OutOfBoundsIndexStopsInDebugBuilds) {
  Arena arena;
  const Tensor tensor = Counting2x3(arena);
  EXPECT_DEATH((void)tensor.at<float>(2, 0), "out of bounds");
  EXPECT_DEATH((void)tensor.at<float>(0, 3), "out of bounds");
  EXPECT_DEATH((void)tensor.at<float>(-1, 0), "out of bounds");
}

TEST(TensorDeathTest, WrongNumberOfIndicesStopsInDebugBuilds) {
  Arena arena;
  const Tensor tensor = Counting2x3(arena);
  EXPECT_DEATH((void)tensor.at<float>(0), "wrong number of indices");
}

TEST(TensorDeathTest, AtWithTheWrongTypeStopsInDebugBuilds) {
  Arena arena;
  const Tensor tensor = Counting2x3(arena);
  EXPECT_DEATH((void)tensor.at<std::int32_t>(0, 0), "wrong element type");
}
#endif
