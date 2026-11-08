#include "ember/safetensors.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "safetensors_writer.h"

namespace {

using ember::DType;
using ember::Safetensors;
using ember::StatusCode;
using ember::testing::RawSafetensors;
using ember::testing::SafetensorsWriter;

std::vector<std::byte> Bytes(std::size_t count) { return std::vector<std::byte>(count); }

// The error message for a file, or "OK".
std::string ParseError(const std::vector<std::byte>& file) {
  const auto parsed = Safetensors::Parse(file);
  if (parsed.ok()) return "OK";
  EXPECT_EQ(parsed.status().code(), StatusCode::kInvalidFormat);
  return parsed.status().message();
}

bool Mentions(const std::string& message, const std::string& part) {
  return message.find(part) != std::string::npos;
}

TEST(Safetensors, ReadsTensorsAndMetadata) {
  SafetensorsWriter writer;
  writer.AddMetadata("format", "pt");
  writer.AddF32("b", {2, 2}, {1, 2, 3, 4});
  writer.AddF16("a", {3}, {0x3C00, 0x4000, 0x4200});
  const std::vector<std::byte> file = writer.Build();

  const auto parsed = Safetensors::Parse(file);
  ASSERT_TRUE(parsed.ok()) << parsed.status().ToString();
  ASSERT_EQ(parsed->tensors().size(), 2U);
  ASSERT_EQ(parsed->metadata().size(), 1U);
  EXPECT_EQ(parsed->metadata()[0].first, "format");
  EXPECT_EQ(parsed->metadata()[0].second, "pt");

  const Safetensors::Entry* b = parsed->Find("b");
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(b->dtype, DType::kF32);
  EXPECT_EQ(b->shape.ToString(), "[2, 2]");
  const auto view = Safetensors::View(*b);
  ASSERT_TRUE(view.ok()) << view.status().ToString();
  EXPECT_EQ(view->at<float>(1, 0), 3.0F);

  const Safetensors::Entry* a = parsed->Find("a");
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(a->dtype, DType::kF16);
  EXPECT_EQ(a->bytes.size(), 6U);
  EXPECT_EQ(parsed->Find("c"), nullptr);
}

TEST(Safetensors, ListsTensorsInByteOrderWhateverTheHeaderOrder) {
  const std::string header = R"({"second":{"dtype":"I8","shape":[2],"data_offsets":[2,4]},)"
                             R"("first":{"dtype":"I8","shape":[2],"data_offsets":[0,2]}})";
  const auto parsed = Safetensors::Parse(RawSafetensors(header, Bytes(4)));
  ASSERT_TRUE(parsed.ok()) << parsed.status().ToString();
  EXPECT_EQ(parsed->tensors()[0].name, "first");
  EXPECT_EQ(parsed->tensors()[1].name, "second");
}

TEST(Safetensors, AnEmptyHeaderObjectIsAnEmptyFile) {
  const auto parsed = Safetensors::Parse(RawSafetensors("{}", {}));
  ASSERT_TRUE(parsed.ok());
  EXPECT_TRUE(parsed->tensors().empty());
}

TEST(Safetensors, RefusesALengthThatDoesNotFit) {
  EXPECT_TRUE(Mentions(ParseError(Bytes(7)), "shorter than"));
  std::vector<std::byte> file = RawSafetensors("{}", {});
  file[0] = std::byte{3};  // says 3 bytes of header, has 2
  EXPECT_TRUE(Mentions(ParseError(file), "past the end"));
  file[7] = std::byte{0xFF};  // a header far over 16 MB
  EXPECT_TRUE(Mentions(ParseError(file), "16 MB"));
}

TEST(Safetensors, RefusesAHeaderThatIsNotAnObject) {
  EXPECT_TRUE(Mentions(ParseError(RawSafetensors("[]", {})), "not a JSON object"));
  EXPECT_TRUE(Mentions(ParseError(RawSafetensors("{", {})), "JSON"));
}

TEST(Safetensors, RefusesBadTensorInfo) {
  const auto with = [](const std::string& info) {
    return ParseError(RawSafetensors(R"({"t":)" + info + "}", Bytes(8)));
  };
  EXPECT_TRUE(Mentions(with(R"(1)"), "not an object"));
  EXPECT_TRUE(Mentions(with(R"({"shape":[2],"data_offsets":[0,8]})"), "no dtype"));
  EXPECT_TRUE(
      Mentions(with(R"({"dtype":"F64","shape":[1],"data_offsets":[0,8]})"), "unsupported dtype"));
  EXPECT_TRUE(Mentions(with(R"({"dtype":"F32","data_offsets":[0,8]})"), "no shape"));
  EXPECT_TRUE(Mentions(with(R"({"dtype":"I8","shape":[1,1,1,1,8],"data_offsets":[0,8]})"),
                       "too many dimensions"));
  EXPECT_TRUE(Mentions(with(R"({"dtype":"I8","shape":[-8],"data_offsets":[0,8]})"), "not a size"));
  EXPECT_TRUE(Mentions(with(R"({"dtype":"I8","shape":[8.5],"data_offsets":[0,8]})"), "not a size"));
  EXPECT_TRUE(Mentions(with(R"({"dtype":"I8","shape":[8]})"), "data_offsets"));
  EXPECT_TRUE(Mentions(with(R"({"dtype":"I8","shape":[8],"data_offsets":[0]})"), "data_offsets"));
  EXPECT_TRUE(
      Mentions(with(R"({"dtype":"I8","shape":[8],"data_offsets":[0,8],"x":1})"), "other than"));
}

TEST(Safetensors, RefusesOffsetsOutsideTheData) {
  const auto with = [](const std::string& offsets) {
    return ParseError(RawSafetensors(
        R"({"t":{"dtype":"I8","shape":[8],"data_offsets":)" + offsets + "}}", Bytes(8)));
  };
  EXPECT_TRUE(Mentions(with("[0,9]"), "outside the file"));
  EXPECT_TRUE(Mentions(with("[8,0]"), "outside the file"));
  EXPECT_TRUE(Mentions(with("[0,18446744073709551615]"), "outside the file"));
  EXPECT_TRUE(Mentions(with("[0,18446744073709551616]"), "outside the file"));
}

TEST(Safetensors, RefusesASizeThatDoesNotMatchTheShape) {
  // 2 f32 values are 8 bytes, not 4.
  const std::string message = ParseError(
      RawSafetensors(R"({"t":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})", Bytes(4)));
  EXPECT_TRUE(Mentions(message, "is 4 bytes, but f32[2] is 8")) << message;
}

TEST(Safetensors, RefusesAShapeWhoseSizeOverflows) {
  // 2^32 * 2^32 wraps to 0 on 64 bits, which would match a zero-byte range.
  const std::string message = ParseError(RawSafetensors(
      R"({"t":{"dtype":"I8","shape":[4294967296,4294967296],"data_offsets":[0,0]}})", {}));
  EXPECT_NE(message, "OK");
}

TEST(Safetensors, RefusesOverlapsGapsAndLeftoverBytes) {
  const auto two = [](const std::string& first, const std::string& second, std::size_t size) {
    return ParseError(RawSafetensors(R"({"a":{"dtype":"I8","shape":[4],"data_offsets":)" + first +
                                         R"(},"b":{"dtype":"I8","shape":[4],"data_offsets":)" +
                                         second + "}}",
                                     Bytes(size)));
  };
  EXPECT_EQ(two("[0,4]", "[4,8]", 8), "OK");
  EXPECT_TRUE(Mentions(two("[0,4]", "[2,6]", 8), "overlaps"));
  EXPECT_TRUE(Mentions(two("[0,4]", "[0,4]", 8), "overlaps"));
  EXPECT_TRUE(Mentions(two("[0,4]", "[5,9]", 9), "gap"));
  EXPECT_TRUE(Mentions(two("[0,4]", "[4,8]", 9), "belong to nothing"));
}

TEST(Safetensors, RefusesMetadataThatIsNotStrings) {
  EXPECT_TRUE(Mentions(ParseError(RawSafetensors(R"({"__metadata__":{"a":1}})", {})), "strings"));
  EXPECT_TRUE(Mentions(ParseError(RawSafetensors(R"({"__metadata__":[]})", {})), "not an object"));
}

TEST(Safetensors, ViewRefusesMisalignedF32) {
  // Pad the header so the data starts one byte past a multiple of 4. A vector's memory is aligned
  // for any basic type, so the data lands on an address no f32 can be read from.
  std::string header = R"({"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
  while ((8 + header.size()) % 4 != 1) header += ' ';
  const std::vector<std::byte> file = RawSafetensors(header, Bytes(4));
  const auto parsed = Safetensors::Parse(file);
  ASSERT_TRUE(parsed.ok()) << parsed.status().ToString();
  ASSERT_NE(reinterpret_cast<std::uintptr_t>(parsed->tensors()[0].bytes.data()) % 4, 0U);
  EXPECT_FALSE(Safetensors::View(parsed->tensors()[0]).ok());
}

}  // namespace
