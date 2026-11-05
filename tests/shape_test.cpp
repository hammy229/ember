#include "ember/shape.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <vector>

namespace {

using ember::Shape;

constexpr std::size_t kSizeMax = std::numeric_limits<std::size_t>::max();

}  // namespace

TEST(Shape, DefaultIsAScalar) {
  const Shape shape;
  EXPECT_EQ(shape.rank(), 0U);
  EXPECT_EQ(shape.element_count(), 1U);
  EXPECT_EQ(shape.ToString(), "[]");
}

TEST(Shape, EmptyListIsAScalar) {
  const auto shape = Shape::Make(std::vector<std::size_t>{});
  ASSERT_TRUE(shape.ok());
  EXPECT_EQ(*shape, Shape{});
}

TEST(Shape, KeepsDimensionsAndCountsElements) {
  const auto shape = Shape::Make({2, 3, 4});
  ASSERT_TRUE(shape.ok());
  EXPECT_EQ(shape->rank(), 3U);
  EXPECT_EQ(shape->dim(0), 2U);
  EXPECT_EQ(shape->dim(1), 3U);
  EXPECT_EQ(shape->dim(2), 4U);
  EXPECT_EQ(shape->element_count(), 24U);
  EXPECT_EQ(shape->ToString(), "[2, 3, 4]");
}

TEST(Shape, AcceptsAVectorOfDimensions) {
  const std::vector<std::size_t> dims{5, 6};
  const auto shape = Shape::Make(dims);
  ASSERT_TRUE(shape.ok());
  EXPECT_EQ(shape->element_count(), 30U);
}

TEST(Shape, ZeroDimensionMeansNoElements) {
  const auto shape = Shape::Make({2, 0, 4});
  ASSERT_TRUE(shape.ok());
  EXPECT_EQ(shape->element_count(), 0U);
  EXPECT_EQ(shape->rank(), 3U);
}

TEST(Shape, AllowsTheMaximumRank) {
  const auto shape = Shape::Make({1, 2, 3, 4});
  ASSERT_TRUE(shape.ok());
  EXPECT_EQ(shape->rank(), Shape::kMaxRank);
}

TEST(Shape, RejectsTooManyDimensions) {
  const auto shape = Shape::Make({1, 2, 3, 4, 5});
  ASSERT_FALSE(shape.ok());
  EXPECT_EQ(shape.status().code(), ember::StatusCode::kInvalidArgument);
}

TEST(Shape, EqualityLooksAtRankAndSizes) {
  EXPECT_EQ(*Shape::Make({2, 3}), *Shape::Make({2, 3}));
  EXPECT_FALSE(*Shape::Make({2, 3}) == *Shape::Make({3, 2}));
  EXPECT_FALSE(*Shape::Make({2, 3}) == *Shape::Make({2, 3, 1}));
}

TEST(Shape, AcceptsExactlyTheLargestCount) {
  const auto shape = Shape::Make({Shape::kMaxElements});
  ASSERT_TRUE(shape.ok());
  EXPECT_EQ(shape->element_count(), Shape::kMaxElements);
}

TEST(Shape, RejectsOneMoreThanTheLargestCount) {
  EXPECT_FALSE(Shape::Make({Shape::kMaxElements + 1}).ok());
}

// These sizes come straight from what a corrupt or hostile model file could claim.
TEST(Shape, RejectsSizesWhoseProductWrapsAround) {
  const std::size_t half_bits = std::size_t{1} << (sizeof(std::size_t) * 4);
  // half_bits * half_bits is exactly 2^bits, which wraps to zero.
  EXPECT_FALSE(Shape::Make({half_bits, half_bits}).ok());
  EXPECT_FALSE(Shape::Make({kSizeMax, 2}).ok());
  EXPECT_FALSE(Shape::Make({2, kSizeMax}).ok());
  // (2^bits - 1)^2 wraps around to exactly 1, which looks like a harmless single element.
  EXPECT_FALSE(Shape::Make({kSizeMax, kSizeMax}).ok());
}

TEST(Shape, ZeroDoesNotHideAnOverflow) {
  EXPECT_FALSE(Shape::Make({0, kSizeMax, kSizeMax}).ok());
  EXPECT_FALSE(Shape::Make({kSizeMax, 0, kSizeMax}).ok());
  EXPECT_FALSE(Shape::Make({kSizeMax, kSizeMax, 0}).ok());
}

TEST(ShapeDeathTest, ReadingPastTheRankIsABug) {
  const auto shape = Shape::Make({2});
  ASSERT_TRUE(shape.ok());
  EXPECT_DEATH((void)shape->dim(1), "past the end of a shape");
}
