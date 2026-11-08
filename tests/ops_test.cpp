#include "ember/ops.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace {

using ember::Arena;
using ember::DType;
using ember::Shape;
using ember::StatusCode;
using ember::Tensor;
using ember::TensorView;

Shape MakeShape(std::initializer_list<std::size_t> dims) { return *Shape::Make(dims); }

Tensor F32(Arena& arena, std::initializer_list<std::size_t> dims, std::vector<float> values) {
  return *Tensor::FromValues<float>(arena, MakeShape(dims), std::span<const float>(values));
}

Tensor Output(Arena& arena, std::initializer_list<std::size_t> dims) {
  return *Tensor::Zeros(arena, DType::kF32, MakeShape(dims));
}

// Reads a tensor out in row by row order, whatever its strides are.
std::vector<float> Values(Arena& arena, const TensorView& tensor) {
  const Tensor copy = *Tensor::CopyOf(arena, tensor);
  const float* first = copy.data<float>();
  return {first, first + copy.element_count()};
}

TEST(Add, AddsElementByElement) {
  Arena arena;
  const Tensor a = F32(arena, {2, 2}, {1, 2, 3, 4});
  const Tensor b = F32(arena, {2, 2}, {10, 20, 30, 40});
  const Tensor out = Output(arena, {2, 2});
  ASSERT_TRUE(ember::Add(a, b, out).ok());
  EXPECT_EQ(Values(arena, out), (std::vector<float>{11, 22, 33, 44}));
}

TEST(Add, FollowsTheStridesOfATransposedInput) {
  Arena arena;
  const Tensor a = F32(arena, {2, 3}, {1, 2, 3, 4, 5, 6});
  const Tensor b = F32(arena, {3, 2}, {10, 40, 20, 50, 30, 60});
  const Tensor out = Output(arena, {3, 2});
  ASSERT_TRUE(ember::Add(*a.Transpose(0, 1), b, out).ok());
  EXPECT_EQ(Values(arena, out), (std::vector<float>{11, 44, 22, 55, 33, 66}));
}

TEST(Add, WritesThroughTheStridesOfAColumnAndLeavesTheRestAlone) {
  Arena arena;
  const Tensor matrix = F32(arena, {2, 2}, {1, 2, 3, 4});
  const Tensor a = F32(arena, {2, 1}, {10, 20});
  const Tensor b = F32(arena, {2, 1}, {1, 1});
  ASSERT_TRUE(ember::Add(a, b, *matrix.Slice(1, 1, 1)).ok());
  EXPECT_EQ(Values(arena, matrix), (std::vector<float>{1, 11, 3, 21}));
}

TEST(Add, WorksOnAScalar) {
  Arena arena;
  const Tensor a = F32(arena, {}, {1.5F});
  const Tensor b = F32(arena, {}, {2.0F});
  const Tensor out = Output(arena, {});
  ASSERT_TRUE(ember::Add(a, b, out).ok());
  EXPECT_FLOAT_EQ(out.at<float>(), 3.5F);
}

TEST(Add, DoesNothingOnAnEmptyTensor) {
  Arena arena;
  const Tensor a = Output(arena, {0, 3});
  const Tensor out = Output(arena, {0, 3});
  EXPECT_TRUE(ember::Add(a, a, out).ok());
}

TEST(Add, RefusesShapesThatDiffer) {
  Arena arena;
  const Tensor a = Output(arena, {2, 3});
  const Tensor b = Output(arena, {3, 2});
  const Tensor out = Output(arena, {2, 3});
  EXPECT_EQ(ember::Add(a, b, out).code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(ember::Add(a, a, b).code(), StatusCode::kInvalidArgument);
}

TEST(Add, RefusesAnythingButFloats) {
  Arena arena;
  const Shape shape = MakeShape({2});
  const Tensor ints = *Tensor::Zeros(arena, DType::kI32, shape);
  const Tensor floats = *Tensor::Zeros(arena, DType::kF32, shape);
  EXPECT_EQ(ember::Add(ints, floats, floats).code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(ember::Add(floats, ints, floats).code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(ember::Add(floats, floats, ints).code(), StatusCode::kInvalidArgument);
}

TEST(Add, CanWriteOverEitherInput) {
  Arena arena;
  const Tensor a = F32(arena, {2}, {1, 2});
  const Tensor b = F32(arena, {2}, {10, 20});
  ASSERT_TRUE(ember::Add(a, b, a).ok());
  EXPECT_EQ(Values(arena, a), (std::vector<float>{11, 22}));
  ASSERT_TRUE(ember::Add(a, b, b).ok());
  EXPECT_EQ(Values(arena, b), (std::vector<float>{21, 42}));
}

TEST(Add, RefusesAnOutputThatPartlyOverlapsAnInput) {
  Arena arena;
  const Tensor line = F32(arena, {4}, {1, 2, 3, 4});
  const Tensor a = *line.Slice(0, 0, 3);
  const Tensor shifted = *line.Slice(0, 1, 3);
  const Tensor other = F32(arena, {3}, {1, 1, 1});
  // Going left to right this would read numbers it had already overwritten.
  EXPECT_EQ(ember::Add(a, other, shifted).code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(Values(arena, line), (std::vector<float>{1, 2, 3, 4}));
}

TEST(Add, RefusesATransposeOfItselfAsTheOutput) {
  Arena arena;
  const Tensor square = F32(arena, {2, 2}, {1, 2, 3, 4});
  const Tensor other = Output(arena, {2, 2});
  // Same memory, same first element, different strides. It only looks like writing in place.
  EXPECT_EQ(ember::Add(square, other, *square.Transpose(0, 1)).code(),
            StatusCode::kInvalidArgument);
}

TEST(Multiply, MultipliesElementByElement) {
  Arena arena;
  const Tensor a = F32(arena, {3}, {1, 2, 3});
  const Tensor b = F32(arena, {3}, {4, 5, 6});
  const Tensor out = Output(arena, {3});
  ASSERT_TRUE(ember::Multiply(a, b, out).ok());
  EXPECT_EQ(Values(arena, out), (std::vector<float>{4, 10, 18}));
}

TEST(Multiply, RefusesShapesThatDiffer) {
  Arena arena;
  const Tensor a = Output(arena, {2});
  const Tensor b = Output(arena, {3});
  EXPECT_EQ(ember::Multiply(a, b, a).code(), StatusCode::kInvalidArgument);
}

TEST(Scale, MultipliesEveryElementByOneNumber) {
  Arena arena;
  const Tensor a = F32(arena, {2, 2}, {1, 2, 3, 4});
  const Tensor out = Output(arena, {2, 2});
  ASSERT_TRUE(ember::Scale(a, 0.5F, out).ok());
  EXPECT_EQ(Values(arena, out), (std::vector<float>{0.5F, 1, 1.5F, 2}));
}

TEST(Scale, CanWriteOverItsInputAndFollowsStrides) {
  Arena arena;
  const Tensor matrix = F32(arena, {2, 2}, {1, 2, 3, 4});
  const Tensor column = *matrix.Slice(1, 0, 1);
  ASSERT_TRUE(ember::Scale(column, 10.0F, column).ok());
  EXPECT_EQ(Values(arena, matrix), (std::vector<float>{10, 2, 30, 4}));
}

TEST(Scale, RefusesABadOutput) {
  Arena arena;
  const Tensor a = Output(arena, {2});
  const Tensor wrong_shape = Output(arena, {3});
  EXPECT_EQ(ember::Scale(a, 2.0F, wrong_shape).code(), StatusCode::kInvalidArgument);
}

TEST(MatMul, MultipliesTwoMatrices) {
  Arena arena;
  const Tensor a = F32(arena, {2, 3}, {1, 2, 3, 4, 5, 6});
  const Tensor b = F32(arena, {3, 2}, {7, 8, 9, 10, 11, 12});
  const Tensor out = Output(arena, {2, 2});
  ASSERT_TRUE(ember::MatMul(a, b, out).ok());
  EXPECT_EQ(Values(arena, out), (std::vector<float>{58, 64, 139, 154}));
}

TEST(MatMul, ReplacesWhatWasInTheOutput) {
  Arena arena;
  const Tensor a = F32(arena, {1, 1}, {2});
  const Tensor b = F32(arena, {1, 1}, {3});
  const Tensor out = F32(arena, {1, 1}, {100});
  ASSERT_TRUE(ember::MatMul(a, b, out).ok());
  EXPECT_EQ(out.at<float>(0, 0), 6.0F);
}

TEST(MatMul, TheIdentityChangesNothing) {
  Arena arena;
  const Tensor a = F32(arena, {2, 2}, {1, 2, 3, 4});
  const Tensor identity = F32(arena, {2, 2}, {1, 0, 0, 1});
  const Tensor out = Output(arena, {2, 2});
  ASSERT_TRUE(ember::MatMul(a, identity, out).ok());
  EXPECT_EQ(Values(arena, out), Values(arena, a));
  ASSERT_TRUE(ember::MatMul(identity, a, out).ok());
  EXPECT_EQ(Values(arena, out), Values(arena, a));
}

// Attention multiplies queries by transposed keys, so this is the case that matters most.
TEST(MatMul, TakesATransposedMatrixOnEitherSide) {
  Arena arena;
  const Tensor a = F32(arena, {2, 3}, {1, 2, 3, 4, 5, 6});
  const Tensor out = Output(arena, {2, 2});
  ASSERT_TRUE(ember::MatMul(a, *a.Transpose(0, 1), out).ok());
  EXPECT_EQ(Values(arena, out), (std::vector<float>{14, 32, 32, 77}));

  const Tensor wide = Output(arena, {3, 3});
  ASSERT_TRUE(ember::MatMul(*a.Transpose(0, 1), a, wide).ok());
  EXPECT_EQ(Values(arena, wide), (std::vector<float>{17, 22, 27, 22, 29, 36, 27, 36, 45}));
}

TEST(MatMul, WritesIntoATransposedOutput) {
  Arena arena;
  const Tensor a = F32(arena, {2, 3}, {1, 2, 3, 4, 5, 6});
  const Tensor b = F32(arena, {3, 1}, {1, 1, 1});
  const Tensor out = Output(arena, {1, 2});
  ASSERT_TRUE(ember::MatMul(a, b, *out.Transpose(0, 1)).ok());
  EXPECT_EQ(Values(arena, out), (std::vector<float>{6, 15}));
}

TEST(MatMul, AnEmptyMiddleDimensionGivesZeros) {
  Arena arena;
  const Tensor a = Output(arena, {2, 0});
  const Tensor b = Output(arena, {0, 2});
  const Tensor out = F32(arena, {2, 2}, {9, 9, 9, 9});
  ASSERT_TRUE(ember::MatMul(a, b, out).ok());
  EXPECT_EQ(Values(arena, out), (std::vector<float>{0, 0, 0, 0}));
}

TEST(MatMul, AnEmptyMiddleDimensionIsFineOnANullPointer) {
  Arena arena;
  const TensorView a = *TensorView::Wrap(nullptr, DType::kF32, MakeShape({2, 0}));
  const TensorView b = *TensorView::Wrap(nullptr, DType::kF32, MakeShape({0, 2}));
  const Tensor out = F32(arena, {2, 2}, {9, 9, 9, 9});
  ASSERT_TRUE(ember::MatMul(a, b, out).ok());
  EXPECT_EQ(Values(arena, out), (std::vector<float>{0, 0, 0, 0}));
}

TEST(MatMul, AnEmptyOutputDoesNothing) {
  Arena arena;
  const Tensor a = Output(arena, {0, 3});
  const Tensor b = Output(arena, {3, 2});
  const Tensor out = Output(arena, {0, 2});
  EXPECT_TRUE(ember::MatMul(a, b, out).ok());
}

TEST(MatMul, RefusesSizesThatDoNotLineUp) {
  Arena arena;
  const Tensor a = Output(arena, {2, 3});
  const Tensor b = Output(arena, {2, 3});
  const Tensor out = Output(arena, {2, 3});
  EXPECT_EQ(ember::MatMul(a, b, out).code(), StatusCode::kInvalidArgument);

  const Tensor b_ok = Output(arena, {3, 4});
  const Tensor wrong_rows = Output(arena, {3, 4});
  const Tensor wrong_columns = Output(arena, {2, 5});
  EXPECT_EQ(ember::MatMul(a, b_ok, wrong_rows).code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(ember::MatMul(a, b_ok, wrong_columns).code(), StatusCode::kInvalidArgument);
}

TEST(MatMul, MultipliesEachMatrixInABatchOnItsOwn) {
  Arena arena;
  const Tensor a = F32(arena, {2, 2, 2}, {1, 0, 0, 1, 2, 0, 0, 2});
  const Tensor b = F32(arena, {2, 2, 1}, {3, 4, 5, 6});
  const Tensor out = Output(arena, {2, 2, 1});
  ASSERT_TRUE(ember::MatMul(a, b, out).ok());
  EXPECT_EQ(Values(arena, out), (std::vector<float>{3, 4, 10, 12}));
}

// The shape attention really uses: [batch, heads, tokens, features] times its own transpose.
TEST(MatMul, HandlesFourDimensionsWithATranspose) {
  Arena arena;
  const Tensor q = F32(arena, {1, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
  const Tensor out = Output(arena, {1, 2, 2, 2});
  ASSERT_TRUE(ember::MatMul(q, *q.Transpose(2, 3), out).ok());
  EXPECT_EQ(Values(arena, out), (std::vector<float>{5, 11, 11, 25, 61, 83, 83, 113}));
}

TEST(MatMul, AnEmptyBatchDoesNothing) {
  Arena arena;
  const TensorView a = *TensorView::Wrap(nullptr, DType::kF32, MakeShape({0, 2, 2}));
  const Tensor out = Output(arena, {0, 2, 2});
  EXPECT_TRUE(ember::MatMul(a, a, out).ok());
}

TEST(MatMul, RefusesBatchesThatDiffer) {
  Arena arena;
  const Tensor a = Output(arena, {2, 2, 2});
  const Tensor b = Output(arena, {3, 2, 2});
  const Tensor out = Output(arena, {2, 2, 2});
  const Tensor wrong_out = Output(arena, {3, 2, 2});
  const Tensor unbatched = Output(arena, {2, 2});
  EXPECT_EQ(ember::MatMul(a, b, out).code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(ember::MatMul(a, a, wrong_out).code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(ember::MatMul(a, unbatched, out).code(), StatusCode::kInvalidArgument);
}

TEST(MatMul, RefusesAnOutputThatTouchesAnInput) {
  Arena arena;
  const Tensor a = F32(arena, {2, 2}, {1, 2, 3, 4});
  const Tensor b = F32(arena, {2, 2}, {5, 6, 7, 8});
  // A matrix product reads each input element many times, so even writing exactly over an input
  // would use numbers it had already replaced.
  EXPECT_EQ(ember::MatMul(a, b, a).code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(ember::MatMul(a, b, b).code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(Values(arena, a), (std::vector<float>{1, 2, 3, 4}));
  EXPECT_EQ(Values(arena, b), (std::vector<float>{5, 6, 7, 8}));
}

// The plainest possible matrix product, written from the definition, to check the real one against.
float TextbookEntry(const TensorView& a, const TensorView& b, std::size_t row, std::size_t column) {
  float total = 0.0F;
  for (std::size_t k = 0; k < a.dim(1); ++k) total += a.at<float>(row, k) * b.at<float>(k, column);
  return total;
}

Tensor RandomMatrix(Arena& arena, std::mt19937& random, std::size_t rows, std::size_t columns) {
  const Tensor matrix = Output(arena, {rows, columns});
  std::uniform_real_distribution<float> value(-10.0F, 10.0F);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < columns; ++j) matrix.at<float>(i, j) = value(random);
  }
  return matrix;
}

// Every mix of plain and transposed inputs and outputs, at random sizes, against the definition.
TEST(MatMul, MatchesTheDefinitionOnRandomMatricesAndLayouts) {
  std::mt19937 random(12345);
  std::uniform_int_distribution<std::size_t> size(1, 40);
  for (int trial = 0; trial < 300; ++trial) {
    Arena arena;
    const std::size_t rows = size(random);
    const std::size_t middle = size(random);
    const std::size_t columns = size(random);
    const bool flip_a = (trial & 1) != 0;
    const bool flip_b = (trial & 2) != 0;
    const bool flip_out = (trial & 4) != 0;

    const Tensor stored_a = flip_a ? RandomMatrix(arena, random, middle, rows)
                                   : RandomMatrix(arena, random, rows, middle);
    const Tensor stored_b = flip_b ? RandomMatrix(arena, random, columns, middle)
                                   : RandomMatrix(arena, random, middle, columns);
    const Tensor stored_out =
        flip_out ? Output(arena, {columns, rows}) : Output(arena, {rows, columns});
    const TensorView a = flip_a ? TensorView(*stored_a.Transpose(0, 1)) : TensorView(stored_a);
    const TensorView b = flip_b ? TensorView(*stored_b.Transpose(0, 1)) : TensorView(stored_b);
    const Tensor out = flip_out ? *stored_out.Transpose(0, 1) : stored_out;

    ASSERT_TRUE(ember::MatMul(a, b, out).ok());
    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < columns; ++j) {
        ASSERT_NEAR(out.at<float>(i, j), TextbookEntry(a, b, i, j), 1e-2F)
            << "trial " << trial << " at " << i << ", " << j;
      }
    }
  }
}

TEST(MatMul, RefusesAVectorOrAScalar) {
  Arena arena;
  const Tensor vector = Output(arena, {2});
  const Tensor matrix = Output(arena, {2, 2});
  const Tensor scalar = Output(arena, {});
  EXPECT_EQ(ember::MatMul(vector, matrix, matrix).code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(ember::MatMul(matrix, vector, matrix).code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(ember::MatMul(matrix, matrix, scalar).code(), StatusCode::kInvalidArgument);
}

TEST(MatMul, RefusesAnythingButFloats) {
  Arena arena;
  const Shape shape = MakeShape({2, 2});
  const Tensor ints = *Tensor::Zeros(arena, DType::kI32, shape);
  const Tensor floats = *Tensor::Zeros(arena, DType::kF32, shape);
  EXPECT_EQ(ember::MatMul(ints, floats, floats).code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(ember::MatMul(floats, ints, floats).code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(ember::MatMul(floats, floats, ints).code(), StatusCode::kInvalidArgument);
}

void ExpectNear(const std::vector<float>& actual, const std::vector<float>& expected,
                float tolerance = 1e-6F) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], tolerance) << "at " << i;
  }
}

TEST(Softmax, GivesTheKnownAnswer) {
  Arena arena;
  const Tensor logits = F32(arena, {3}, {1, 2, 3});
  const Tensor out = Output(arena, {3});
  ASSERT_TRUE(ember::Softmax(logits, 0, out).ok());
  ExpectNear(Values(arena, out), {0.09003057F, 0.24472847F, 0.66524096F});
}

TEST(Softmax, EqualLogitsGiveEqualChances) {
  Arena arena;
  const Tensor logits = F32(arena, {4}, {7, 7, 7, 7});
  const Tensor out = Output(arena, {4});
  ASSERT_TRUE(ember::Softmax(logits, 0, out).ok());
  ExpectNear(Values(arena, out), {0.25F, 0.25F, 0.25F, 0.25F});
}

TEST(Softmax, WorksOnEachRowOrEachColumn) {
  Arena arena;
  const Tensor logits = F32(arena, {2, 2}, {0, 0, 1, 3});
  const Tensor out = Output(arena, {2, 2});
  ASSERT_TRUE(ember::Softmax(logits, 1, out).ok());
  ExpectNear(Values(arena, out), {0.5F, 0.5F, 0.11920292F, 0.88079708F});
  ASSERT_TRUE(ember::Softmax(logits, 0, out).ok());
  ExpectNear(Values(arena, out), {0.26894142F, 0.04742587F, 0.73105858F, 0.95257413F});
}

TEST(Softmax, AxisAndTransposeAgree) {
  Arena arena;
  const Tensor logits = F32(arena, {2, 3}, {1, -2, 3, 0.5F, 4, -1});
  const Tensor by_axis = Output(arena, {2, 3});
  const Tensor by_view = Output(arena, {3, 2});
  ASSERT_TRUE(ember::Softmax(logits, 0, by_axis).ok());
  ASSERT_TRUE(ember::Softmax(*logits.Transpose(0, 1), 1, by_view).ok());
  ExpectNear(Values(arena, by_axis), Values(arena, *by_view.Transpose(0, 1)));
}

TEST(Softmax, WorksInTheMiddleOfFourDimensions) {
  Arena arena;
  const Tensor logits = F32(arena, {1, 2, 2, 1}, {0, 0, 1, 3});
  const Tensor out = Output(arena, {1, 2, 2, 1});
  ASSERT_TRUE(ember::Softmax(logits, 2, out).ok());
  ExpectNear(Values(arena, out), {0.5F, 0.5F, 0.11920292F, 0.88079708F});
}

TEST(Softmax, CanWriteOverItsInput) {
  Arena arena;
  const Tensor logits = F32(arena, {3}, {1, 2, 3});
  ASSERT_TRUE(ember::Softmax(logits, 0, logits).ok());
  ExpectNear(Values(arena, logits), {0.09003057F, 0.24472847F, 0.66524096F});
}

// e^1000 does not fit in a float, so the plain formula gives infinity over infinity here.
TEST(Softmax, HugeLogitsDoNotTurnIntoNotANumber) {
  Arena arena;
  const Tensor logits = F32(arena, {3}, {1001, 1002, 1003});
  const Tensor out = Output(arena, {3});
  ASSERT_TRUE(ember::Softmax(logits, 0, out).ok());
  ExpectNear(Values(arena, out), {0.09003057F, 0.24472847F, 0.66524096F}, 1e-5F);
}

// The other way around, every e^x rounds to zero and the plain formula divides zero by zero.
TEST(Softmax, VeryNegativeLogitsDoNotTurnIntoNotANumber) {
  Arena arena;
  const Tensor logits = F32(arena, {3}, {-1003, -1002, -1001});
  const Tensor out = Output(arena, {3});
  ASSERT_TRUE(ember::Softmax(logits, 0, out).ok());
  ExpectNear(Values(arena, out), {0.09003057F, 0.24472847F, 0.66524096F}, 1e-5F);
}

// Attention hides future tokens by setting their score to negative infinity.
TEST(Softmax, NegativeInfinityGetsExactlyZero) {
  Arena arena;
  const float masked = -std::numeric_limits<float>::infinity();
  const Tensor logits = F32(arena, {3}, {0, masked, 0});
  const Tensor out = Output(arena, {3});
  ASSERT_TRUE(ember::Softmax(logits, 0, out).ok());
  EXPECT_EQ(Values(arena, out), (std::vector<float>{0.5F, 0.0F, 0.5F}));
}

TEST(Softmax, TheFullRangeOfFloatsDoesNotBreakIt) {
  Arena arena;
  const float top = std::numeric_limits<float>::max();
  const Tensor logits = F32(arena, {2}, {-top, top});
  const Tensor out = Output(arena, {2});
  ASSERT_TRUE(ember::Softmax(logits, 0, out).ok());
  EXPECT_EQ(Values(arena, out), (std::vector<float>{0.0F, 1.0F}));
}

// A long row of tiny chances, like a large vocabulary. Summing a million of these in a float loses
// a visible share of the total to rounding.
TEST(Softmax, ALongRowOfSmallValuesStaysAccurate) {
  Arena arena;
  const std::size_t count = 1000000;
  std::vector<float> values(count, -10.0F);
  values[0] = 0.0F;
  const Tensor logits =
      *Tensor::FromValues<float>(arena, MakeShape({count}), std::span<const float>(values));
  const Tensor out = Output(arena, {count});
  ASSERT_TRUE(ember::Softmax(logits, 0, out).ok());
  const double tiny = std::exp(-10.0);
  const double expected_top = 1.0 / (1.0 + static_cast<double>(count - 1) * tiny);
  EXPECT_NEAR(out.at<float>(0), expected_top, expected_top * 1e-5);
}

TEST(Softmax, RefusesARowItCannotTurnIntoChances) {
  Arena arena;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  const std::vector<std::vector<float>> bad_rows = {
      {nan, 1, 2},  // First, where std::max would keep it.
      {1, nan, 2},  // Later, where std::max would quietly drop it.
      {1, 2, nan},
      {1, infinity, 2},
      {-infinity, -infinity, -infinity},  // Every token hidden, so nothing is left to choose.
  };
  for (const std::vector<float>& row : bad_rows) {
    const Tensor logits = F32(arena, {3}, row);
    const Tensor out = Output(arena, {3});
    EXPECT_EQ(ember::Softmax(logits, 0, out).code(), StatusCode::kInvalidArgument);
  }
}

TEST(Softmax, FindsABadValueInALaterRow) {
  Arena arena;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const Tensor logits = F32(arena, {2, 2}, {1, 2, 3, nan});
  const Tensor out = Output(arena, {2, 2});
  EXPECT_EQ(ember::Softmax(logits, 1, out).code(), StatusCode::kInvalidArgument);
}

TEST(Softmax, AnEmptyTensorDoesNothing) {
  Arena arena;
  const TensorView logits = *TensorView::Wrap(nullptr, DType::kF32, MakeShape({0, 3}));
  const Tensor out = Output(arena, {0, 3});
  EXPECT_TRUE(ember::Softmax(logits, 1, out).ok());
}

// A zero in the last dimension leaves the outer loops running, so these would move a null pointer
// if the empty check were missing. Only the sanitize build can see that, and CI runs it.
TEST(EmptyTensors, ANullPointerIsNeverMovedWhenTheOuterDimensionHasEntries) {
  const Tensor empty = *Tensor::Wrap(nullptr, DType::kF32, MakeShape({3, 0}));
  EXPECT_TRUE(ember::Add(empty, empty, empty).ok());
  EXPECT_TRUE(ember::Multiply(empty, empty, empty).ok());
  EXPECT_TRUE(ember::Scale(empty, 2.0F, empty).ok());
  EXPECT_TRUE(ember::Softmax(empty, 0, empty).ok());

  Arena arena;
  const TensorView a = *TensorView::Wrap(nullptr, DType::kF32, MakeShape({3, 0, 2}));
  const Tensor b = Output(arena, {3, 2, 2});
  const Tensor out = *Tensor::Wrap(nullptr, DType::kF32, MakeShape({3, 0, 2}));
  EXPECT_TRUE(ember::MatMul(a, b, out).ok());

  const TensorView batched_a = *TensorView::Wrap(nullptr, DType::kF32, MakeShape({2, 3, 0}));
  const TensorView batched_b = *TensorView::Wrap(nullptr, DType::kF32, MakeShape({2, 0, 2}));
  const Tensor filled = Output(arena, {2, 3, 2});
  filled.Fill(9.0F);
  ASSERT_TRUE(ember::MatMul(batched_a, batched_b, filled).ok());
  EXPECT_EQ(Values(arena, filled), std::vector<float>(12, 0.0F));
}

TEST(Softmax, CanWriteOverEachRowOrColumnOfAMatrix) {
  for (const std::size_t axis : {std::size_t{0}, std::size_t{1}}) {
    Arena arena;
    const Tensor logits = F32(arena, {2, 3}, {1, -2, 3, 0.5F, 4, -1});
    const Tensor apart = Output(arena, {2, 3});
    ASSERT_TRUE(ember::Softmax(logits, axis, apart).ok());
    ASSERT_TRUE(ember::Softmax(logits, axis, logits).ok());
    ExpectNear(Values(arena, logits), Values(arena, apart));
  }
}

TEST(Softmax, CanWriteOverATransposedInput) {
  Arena arena;
  const Tensor stored = F32(arena, {2, 3}, {1, -2, 3, 0.5F, 4, -1});
  const Tensor flipped = *stored.Transpose(0, 1);
  const Tensor apart = Output(arena, {3, 2});
  ASSERT_TRUE(ember::Softmax(flipped, 1, apart).ok());
  ASSERT_TRUE(ember::Softmax(flipped, 1, flipped).ok());
  ExpectNear(Values(arena, flipped), Values(arena, apart));
}

TEST(Add, CanWriteOverATransposedInput) {
  Arena arena;
  const Tensor stored = F32(arena, {2, 3}, {1, 2, 3, 4, 5, 6});
  const Tensor flipped = *stored.Transpose(0, 1);
  const Tensor other = F32(arena, {3, 2}, {10, 40, 20, 50, 30, 60});
  ASSERT_TRUE(ember::Add(flipped, other, flipped).ok());
  EXPECT_EQ(Values(arena, flipped), (std::vector<float>{11, 44, 22, 55, 33, 66}));
}

// Attention stores [tokens, heads, features] and swaps the first two, so the batch dimension of the
// product has a stride that is not the plain row by row one.
TEST(MatMul, HandlesASwappedBatchDimension) {
  Arena arena;
  // Two tokens and two heads. Each head has a 1x2 row of features.
  const Tensor stored = F32(arena, {2, 2, 1, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
  const Tensor by_head = *stored.Transpose(0, 1);
  const Tensor b = F32(arena, {2, 2, 2, 1}, {1, 1, 1, 0, 0, 1, 2, 2});
  const Tensor out = Output(arena, {2, 2, 1, 1});
  ASSERT_TRUE(ember::MatMul(by_head, b, out).ok());
  // by_head[0][0] is [1, 2], [0][1] is [5, 6], [1][0] is [3, 4], and [1][1] is [7, 8].
  EXPECT_EQ(Values(arena, out), (std::vector<float>{3, 5, 4, 30}));
}

TEST(Softmax, RefusesABadAxisShapeOrType) {
  Arena arena;
  const Tensor logits = Output(arena, {2, 3});
  const Tensor out = Output(arena, {2, 3});
  const Tensor wrong_shape = Output(arena, {3, 2});
  const Tensor ints = *Tensor::Zeros(arena, DType::kI32, MakeShape({2, 3}));
  const Tensor scalar = Output(arena, {});
  EXPECT_EQ(ember::Softmax(logits, 2, out).code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(ember::Softmax(logits, 0, wrong_shape).code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(ember::Softmax(ints, 0, out).code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(ember::Softmax(logits, 0, ints).code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(ember::Softmax(scalar, 0, scalar).code(), StatusCode::kInvalidArgument);
}

TEST(Softmax, RefusesAnOutputThatPartlyOverlapsItsInput) {
  Arena arena;
  const Tensor line = F32(arena, {4}, {1, 2, 3, 4});
  EXPECT_EQ(ember::Softmax(*line.Slice(0, 0, 3), 0, *line.Slice(0, 1, 3)).code(),
            StatusCode::kInvalidArgument);
}

}  // namespace

namespace {

TEST(Silu, MatchesItsDefinitionAndStaysFiniteAtTheExtremes) {
  Arena arena;
  const std::vector<float> inputs{-1000.0F, -5.0F, -1.0F, 0.0F, 0.5F, 3.0F, 1000.0F};
  const Tensor a = F32(arena, {inputs.size()}, inputs);
  const Tensor out = Output(arena, {inputs.size()});
  ASSERT_TRUE(ember::Silu(a, out).ok());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const double x = inputs[i];
    EXPECT_NEAR(out.at<float>(i), x / (1.0 + std::exp(-x)), 1e-6 * std::max(1.0, std::abs(x)));
  }
  EXPECT_EQ(out.at<float>(6), 1000.0F);
  ASSERT_TRUE(ember::Silu(a, a).ok());
  EXPECT_EQ(Values(arena, a), Values(arena, out));
}

}  // namespace
