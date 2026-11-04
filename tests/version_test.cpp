#include <gtest/gtest.h>

#include <string_view>

#include "ember/version.h"

TEST(Version, MatchesTheProjectVersion) {
  EXPECT_EQ(std::string_view(ember::version()), EMBER_PROJECT_VERSION);
}
