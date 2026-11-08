// Writes starting inputs for the fuzzers: valid files they can mutate, so they spend their time on
// the checks after parsing instead of first having to discover the format by chance.
//
// usage: ember_fuzz_seeds DIRECTORY

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "../tests/tiny_llama.h"

namespace {

void Write(const std::filesystem::path& path, const std::vector<std::byte>& bytes) {
  std::ofstream(path, std::ios::binary)
      .write(reinterpret_cast<const char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: ember_fuzz_seeds DIRECTORY\n";
    return 2;
  }
  const std::filesystem::path root = argv[1];
  std::filesystem::create_directories(root / "safetensors");
  std::filesystem::create_directories(root / "json");

  ember::testing::TinyLlamaOptions options;
  options.layers = 1;
  options.tie = true;
  Write(root / "safetensors" / "tiny_f32", ember::testing::TinyLlamaWeights(options));
  options.f16 = true;
  Write(root / "safetensors" / "tiny_f16", ember::testing::TinyLlamaWeights(options));

  std::ofstream(root / "json" / "config") << ember::testing::TinyLlamaConfig(options);
  std::ofstream(root / "json" / "escapes") << R"({"a": ["é😀", -1.5e3, true, null]})";
  return 0;
}
