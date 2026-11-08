// Breaks a valid model file in many ways and checks that loading it fails cleanly every time. On
// its own this proves no corrupt file crashes the loader. Under the sanitizer build it also proves
// none of them makes it read out of bounds or hit undefined behavior, which is the real point.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "ember/llama.h"
#include "tiny_llama.h"

namespace {

using ember::LlamaModel;
using ember::testing::TinyLlamaConfig;
using ember::testing::TinyLlamaOptions;
using ember::testing::TinyLlamaWeights;

// One layer keeps the file small enough to try every truncation.
TinyLlamaOptions SmallModel() {
  TinyLlamaOptions options;
  options.layers = 1;
  options.f16 = true;
  return options;
}

std::size_t HeaderEnd(const std::vector<std::byte>& file) {
  std::uint64_t length = 0;
  for (std::size_t i = 8; i > 0; --i)
    length = (length << 8) | static_cast<std::uint64_t>(file[i - 1]);
  return 8 + static_cast<std::size_t>(length);
}

// Loading must give a model or an error. Reaching the end of this function is the test.
void LoadOrFail(const std::string& config, std::span<const std::byte> file) {
  const auto model = LlamaModel::FromBytes(config, file);
  if (model.ok()) {
    // A file that still loads must still give weights of the configured shapes.
    EXPECT_EQ((*model)->weights().layers.size(), 1U);
  } else {
    EXPECT_FALSE(model.status().message().empty());
  }
}

TEST(LoaderCorruption, TheUntouchedFileLoads) {
  const std::vector<std::byte> file = TinyLlamaWeights(SmallModel());
  EXPECT_TRUE(LlamaModel::FromBytes(TinyLlamaConfig(SmallModel()), file).ok());
}

TEST(LoaderCorruption, EveryTruncationIsRefused) {
  const std::string config = TinyLlamaConfig(SmallModel());
  const std::vector<std::byte> file = TinyLlamaWeights(SmallModel());
  for (std::size_t size = 0; size < file.size(); ++size) {
    // A copy of exactly this size, so reading one byte past it is a sanitizer error.
    const std::vector<std::byte> cut(file.begin(),
                                     file.begin() + static_cast<std::ptrdiff_t>(size));
    EXPECT_FALSE(LlamaModel::FromBytes(config, cut).ok()) << "size " << size;
  }
}

TEST(LoaderCorruption, RandomByteChangesInTheHeaderNeverCrash) {
  const std::string config = TinyLlamaConfig(SmallModel());
  const std::vector<std::byte> file = TinyLlamaWeights(SmallModel());
  const std::size_t header_end = HeaderEnd(file);
  std::mt19937 random(7);
  std::uniform_int_distribution<std::size_t> position(0, header_end - 1);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_int_distribution<int> edits(1, 4);
  for (int trial = 0; trial < 5000; ++trial) {
    std::vector<std::byte> broken = file;
    for (int i = edits(random); i > 0; --i) broken[position(random)] = std::byte(byte(random));
    LoadOrFail(config, broken);
  }
}

TEST(LoaderCorruption, DigitsSwappedInTheHeaderNeverCrash) {
  // Changing one digit to another keeps the JSON valid, so these reach the offset and shape checks
  // instead of stopping at the parser.
  const std::string config = TinyLlamaConfig(SmallModel());
  const std::vector<std::byte> file = TinyLlamaWeights(SmallModel());
  const std::size_t header_end = HeaderEnd(file);
  std::size_t tried = 0;
  for (std::size_t i = 8; i < header_end; ++i) {
    const auto c = static_cast<char>(file[i]);
    if (c < '0' || c > '9') continue;
    for (char digit = '0'; digit <= '9'; ++digit) {
      if (digit == c) continue;
      std::vector<std::byte> broken = file;
      broken[i] = static_cast<std::byte>(digit);
      LoadOrFail(config, broken);
      ++tried;
    }
  }
  EXPECT_GT(tried, 100U);
}

TEST(LoaderCorruption, ChangesInTheWeightsStillLoad) {
  // The bytes after the header are just numbers. Any bits there are some f16 value, so the file
  // stays valid. Whether a NaN weight is caught is the forward pass's job, not the loader's.
  const std::string config = TinyLlamaConfig(SmallModel());
  std::vector<std::byte> file = TinyLlamaWeights(SmallModel());
  for (std::size_t i = HeaderEnd(file); i < file.size(); i += 7) file[i] = std::byte{0xFF};
  EXPECT_TRUE(LlamaModel::FromBytes(config, file).ok());
}

TEST(LoaderCorruption, RandomConfigEditsNeverCrash) {
  const std::string config = TinyLlamaConfig(SmallModel());
  const std::vector<std::byte> file = TinyLlamaWeights(SmallModel());
  std::mt19937 random(11);
  std::uniform_int_distribution<std::size_t> position(0, config.size() - 1);
  std::uniform_int_distribution<int> byte(0, 127);
  for (int trial = 0; trial < 3000; ++trial) {
    std::string broken = config;
    broken[position(random)] = static_cast<char>(byte(random));
    LoadOrFail(broken, file);
  }
}

}  // namespace
