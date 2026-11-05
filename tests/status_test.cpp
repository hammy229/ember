#include "ember/status.h"

#include <gtest/gtest.h>

#include <string>

TEST(Status, DefaultIsOk) {
  ember::Status s;
  EXPECT_TRUE(s.ok());
  EXPECT_EQ(s.code(), ember::StatusCode::kOk);
  EXPECT_EQ(s.message(), "");
}

TEST(Status, OkFactoryMatchesDefault) { EXPECT_TRUE(ember::Status::Ok().ok()); }

TEST(Status, NotFoundKeepsCodeAndMessage) {
  auto s = ember::Status::NotFound("weights.bin is missing");
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), ember::StatusCode::kNotFound);
  EXPECT_EQ(s.message(), "weights.bin is missing");
}

TEST(Status, InvalidFormatKeepsCodeAndMessage) {
  auto s = ember::Status::InvalidFormat("bad magic number");
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), ember::StatusCode::kInvalidFormat);
  EXPECT_EQ(s.message(), "bad magic number");
}

TEST(Status, InvalidArgumentKeepsCodeAndMessage) {
  auto s = ember::Status::InvalidArgument("negative size");
  EXPECT_EQ(s.code(), ember::StatusCode::kInvalidArgument);
}

TEST(Status, OutOfMemoryKeepsCodeAndMessage) {
  auto s = ember::Status::OutOfMemory("arena is full");
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), ember::StatusCode::kOutOfMemory);
  EXPECT_EQ(s.ToString(), "OUT_OF_MEMORY: arena is full");
}

TEST(Status, ToStringNamesTheCodeAndMessage) {
  EXPECT_EQ(ember::Status::Ok().ToString(), "OK");
  EXPECT_EQ(ember::Status::NotFound("a.bin").ToString(), "NOT_FOUND: a.bin");
  EXPECT_EQ(ember::Status::InvalidFormat("bad header").ToString(), "INVALID_FORMAT: bad header");
  EXPECT_EQ(ember::Status::InvalidArgument("x").ToString(), "INVALID_ARGUMENT: x");
}

TEST(Status, EmptyMessageIsAllowed) {
  EXPECT_EQ(ember::Status::NotFound("").ToString(), "NOT_FOUND");
}

TEST(Status, KeepsUnicodeAndLongMessages) {
  std::string long_msg(100000, 'x');
  EXPECT_EQ(ember::Status::InvalidFormat(long_msg).message().size(), 100000U);
  EXPECT_EQ(ember::Status::NotFound("modèle.bin").message(), "modèle.bin");
}

TEST(Status, CopiesAreIndependent) {
  auto a = ember::Status::NotFound("a");
  auto b = a;
  b = ember::Status::Ok();
  EXPECT_FALSE(a.ok());
  EXPECT_TRUE(b.ok());
}
