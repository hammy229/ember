#include "ember/dtype.h"

#include <gtest/gtest.h>

#include <cstdint>

TEST(DType, SizesAndNames) {
  EXPECT_EQ(ember::SizeOf(ember::DType::kF32), 4U);
  EXPECT_EQ(ember::SizeOf(ember::DType::kI32), 4U);
  EXPECT_EQ(ember::SizeOf(ember::DType::kI8), 1U);
  EXPECT_EQ(ember::Name(ember::DType::kF32), "f32");
  EXPECT_EQ(ember::Name(ember::DType::kI32), "i32");
  EXPECT_EQ(ember::Name(ember::DType::kI8), "i8");
}

TEST(DType, MapsCppTypesToDTypes) {
  static_assert(ember::kDTypeOf<float> == ember::DType::kF32);
  static_assert(ember::kDTypeOf<std::int32_t> == ember::DType::kI32);
  static_assert(ember::kDTypeOf<std::int8_t> == ember::DType::kI8);
  static_assert(!ember::TensorElement<double>);
  static_assert(!ember::TensorElement<int*>);
}
