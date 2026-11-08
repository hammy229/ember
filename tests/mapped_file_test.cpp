#include "ember/mapped_file.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

namespace {

using ember::MappedFile;
using ember::StatusCode;

// A file in the system temp directory that deletes itself.
class TempFile {
 public:
  explicit TempFile(const std::string& contents) {
    path_ = std::filesystem::temp_directory_path() /
            ("ember_mapped_file_test_" + std::to_string(counter_++) + ".dat");
    std::ofstream out(path_, std::ios::binary);
    out << contents;
  }
  ~TempFile() { std::filesystem::remove(path_); }
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;
  std::string path() const { return path_.string(); }

 private:
  static inline int counter_ = 0;
  std::filesystem::path path_;
};

std::string Text(const MappedFile& file) {
  return {reinterpret_cast<const char*>(file.bytes().data()), file.size()};
}

TEST(MappedFile, MapsTheWholeFile) {
  const TempFile temp("hello, weights");
  const auto file = MappedFile::Open(temp.path());
  ASSERT_TRUE(file.ok()) << file.status().ToString();
  EXPECT_EQ(Text(*file), "hello, weights");
}

TEST(MappedFile, OpensAnEmptyFileWithNoBytes) {
  const TempFile temp("");
  const auto file = MappedFile::Open(temp.path());
  ASSERT_TRUE(file.ok()) << file.status().ToString();
  EXPECT_EQ(file->size(), 0U);
  EXPECT_TRUE(file->bytes().empty());
}

TEST(MappedFile, AMissingFileIsNotFound) {
  const auto file = MappedFile::Open("/definitely/not/a/real/ember/file");
  ASSERT_FALSE(file.ok());
  EXPECT_EQ(file.status().code(), StatusCode::kNotFound);
}

TEST(MappedFile, ADirectoryIsRefused) {
  const auto file = MappedFile::Open(std::filesystem::temp_directory_path().string());
  ASSERT_FALSE(file.ok());
  EXPECT_EQ(file.status().code(), StatusCode::kInvalidArgument);
}

TEST(MappedFile, MovingHandsOverTheMapping) {
  const TempFile temp("abc");
  auto opened = MappedFile::Open(temp.path());
  ASSERT_TRUE(opened.ok());
  MappedFile first = std::move(opened).value();
  MappedFile second = std::move(first);
  EXPECT_EQ(Text(second), "abc");
  EXPECT_EQ(first.size(), 0U);  // NOLINT(bugprone-use-after-move): checking the moved-from state

  MappedFile third;
  third = std::move(second);
  EXPECT_EQ(Text(third), "abc");
  EXPECT_TRUE(second.bytes().empty());  // NOLINT(bugprone-use-after-move)
}

}  // namespace
