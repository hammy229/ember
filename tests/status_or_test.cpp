#include "ember/status_or.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>

namespace {

ember::StatusOr<int> Half(int value) {
  if (value % 2 != 0) return ember::Status::InvalidArgument("odd");
  return value / 2;
}

ember::StatusOr<int> Quarter(int value) {
  EMBER_ASSIGN_OR_RETURN(const int half, Half(value));
  EMBER_ASSIGN_OR_RETURN(const int quarter, Half(half));
  return quarter;
}

ember::Status RequireEven(int value) {
  if (value % 2 != 0) return ember::Status::InvalidArgument("odd");
  return ember::Status::Ok();
}

ember::Status RequireBothEven(int a, int b) {
  EMBER_RETURN_IF_ERROR(RequireEven(a));
  EMBER_RETURN_IF_ERROR(RequireEven(b));
  return ember::Status::Ok();
}

}  // namespace

TEST(StatusOr, HoldsAValue) {
  const ember::StatusOr<std::string> s(std::string("tensor"));
  ASSERT_TRUE(s.ok());
  EXPECT_TRUE(s.status().ok());
  EXPECT_EQ(s.value(), "tensor");
  EXPECT_EQ(*s, "tensor");
  EXPECT_EQ(s->size(), 6U);
}

TEST(StatusOr, HoldsAnError) {
  const ember::StatusOr<int> s(ember::Status::NotFound("weights.bin"));
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.status().code(), ember::StatusCode::kNotFound);
  EXPECT_EQ(s.status().message(), "weights.bin");
}

TEST(StatusOr, ValueCanBeChanged) {
  ember::StatusOr<int> s(1);
  *s = 5;
  EXPECT_EQ(s.value(), 5);
}

TEST(StatusOr, MovesOutTypesThatCannotBeCopied) {
  ember::StatusOr<std::unique_ptr<int>> s(std::make_unique<int>(7));
  const std::unique_ptr<int> taken = std::move(s).value();
  EXPECT_EQ(*taken, 7);
}

TEST(StatusOr, AssignOrReturnPassesTheValueThrough) {
  const auto quarter = Quarter(12);
  ASSERT_TRUE(quarter.ok());
  EXPECT_EQ(*quarter, 3);
}

TEST(StatusOr, AssignOrReturnStopsAtTheFirstError) {
  EXPECT_EQ(Quarter(7).status().code(), ember::StatusCode::kInvalidArgument);
  EXPECT_EQ(Quarter(6).status().code(), ember::StatusCode::kInvalidArgument);
}

TEST(StatusOr, ReturnIfErrorStopsAtTheFirstError) {
  EXPECT_TRUE(RequireBothEven(2, 4).ok());
  EXPECT_FALSE(RequireBothEven(1, 4).ok());
  EXPECT_FALSE(RequireBothEven(2, 3).ok());
}

TEST(StatusOrDeathTest, ReadingAFailedValueStops) {
  const ember::StatusOr<int> s(ember::Status::NotFound("gone"));
  EXPECT_DEATH((void)s.value(), "failed StatusOr");
}

TEST(StatusOrDeathTest, BuildingFromAnOkStatusStops) {
  EXPECT_DEATH((void)ember::StatusOr<int>{ember::Status::Ok()}, "must hold an error");
}
