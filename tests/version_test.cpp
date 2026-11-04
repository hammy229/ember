#include "ember/version.h"

#include <gtest/gtest.h>

#include <string_view>

TEST(Version, MatchesTheProjectVersion) {
  EXPECT_EQ(std::string_view(ember::version()), EMBER_PROJECT_VERSION);
}
