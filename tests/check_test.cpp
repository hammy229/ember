#include "ember/check.h"

#include <gtest/gtest.h>

TEST(Check, PassingCheckDoesNothing) {
  int runs = 0;
  EMBER_CHECK(++runs == 1, "should pass");
  EXPECT_EQ(runs, 1);
}

TEST(CheckDeathTest, FailingCheckNamesTheConditionAndReason) {
  const int two = 2;
  EXPECT_DEATH(EMBER_CHECK(two == 3, "two is not three"), "two == 3: two is not three");
}

TEST(CheckDeathTest, FailingCheckNamesTheFile) {
  EXPECT_DEATH(EMBER_CHECK(false, "stop"), "check_test.cpp:[0-9]+");
}

#ifdef NDEBUG
TEST(Check, DebugCheckNeverRunsInRelease) {
  int runs = 0;
  EMBER_DCHECK(++runs == 100, "must not run");
  EXPECT_EQ(runs, 0);
}
#else
TEST(CheckDeathTest, DebugCheckStopsInDebugBuilds) {
  EXPECT_DEATH(EMBER_DCHECK(false, "stop in debug"), "stop in debug");
}
#endif
