#include "ember/ops.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
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

TEST(Scale, RefusesABadOutput) {
  Arena arena;
  const Tensor a = Output(arena, {2});
  const Tensor wrong_shape = Output(arena, {3});
  EXPECT_EQ(ember::Scale(a, 2.0F, wrong_shape).code(), StatusCode::kInvalidArgument);
}

}  // namespace
