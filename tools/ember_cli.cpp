#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "ember/forward.h"
#include "ember/llama.h"
#include "ember/timer.h"
#include "ember/version.h"

namespace {

int Usage() {
  std::cerr << "usage: ember                    print the version\n"
               "       ember inspect MODEL_DIR  load a model and describe it\n"
               "       ember logits MODEL_DIR [--out FILE] TOKEN_ID...\n"
               "                                run the model and show the likeliest next tokens;\n"
               "                                --out saves every logit as little-endian f32\n";
  return 2;
}

int Inspect(const std::string& directory) {
  const ember::Stopwatch stopwatch;
  const auto model = ember::LlamaModel::Load(directory);
  const double load_ms = stopwatch.ElapsedMs();
  if (!model.ok()) {
    std::cerr << "error: " << model.status().ToString() << '\n';
    return 1;
  }
  const ember::LlamaConfig& c = (*model)->config();
  std::cout << "layers " << c.layer_count << ", hidden " << c.hidden_size << ", heads "
            << c.head_count << " (" << c.kv_head_count << " key/value), head_dim " << c.head_dim
            << ", mlp " << c.intermediate_size << ", vocab " << c.vocab_size << ", context "
            << c.max_positions << (c.tie_word_embeddings ? ", tied embeddings" : "") << '\n'
            << "parameters " << (*model)->parameter_count() << '\n'
            << "widened from f16 " << (*model)->widened_bytes() << " bytes\n"
            << "loaded in " << load_ms << " ms\n";
  return 0;
}

int Logits(const std::string& directory, const std::string& out_path,
           const std::vector<std::int32_t>& tokens) {
  const auto model = ember::LlamaModel::Load(directory);
  if (!model.ok()) {
    std::cerr << "error: " << model.status().ToString() << '\n';
    return 1;
  }
  ember::Arena arena;
  const ember::Stopwatch stopwatch;
  const auto logits = ember::Forward(**model, tokens, arena);
  const double forward_ms = stopwatch.ElapsedMs();
  if (!logits.ok()) {
    std::cerr << "error: " << logits.status().ToString() << '\n';
    return 1;
  }
  const std::size_t vocab = logits->dim(1);
  const float* last = logits->data<float>() + (tokens.size() - 1) * vocab;
  std::vector<std::size_t> order(vocab);
  std::iota(order.begin(), order.end(), std::size_t{0});
  const std::size_t shown = std::min<std::size_t>(5, vocab);
  std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(shown), order.end(),
                    [&](std::size_t a, std::size_t b) { return last[a] > last[b]; });
  std::cout << "likeliest next tokens after " << tokens.size() << " tokens:\n";
  for (std::size_t i = 0; i < shown; ++i) {
    std::cout << "  " << order[i] << "  logit " << last[order[i]] << '\n';
  }
  std::cout << "forward pass " << forward_ms << " ms\n";
  if (!out_path.empty()) {
    std::ofstream out(out_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(logits->data<float>()),
              static_cast<std::streamsize>(logits->byte_size()));
    if (!out) {
      std::cerr << "error: could not write " << out_path << '\n';
      return 1;
    }
  }
  return 0;
}

int RunLogits(int argc, char** argv) {
  if (argc < 4) return Usage();
  const std::string directory = argv[2];
  std::string out_path;
  int first = 3;
  if (std::string_view(argv[3]) == "--out") {
    if (argc < 6) return Usage();
    out_path = argv[4];
    first = 5;
  }
  std::vector<std::int32_t> tokens;
  for (int i = first; i < argc; ++i) {
    const std::string_view text = argv[i];
    std::int32_t token = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), token);
    if (error != std::errc() || end != text.data() + text.size()) return Usage();
    tokens.push_back(token);
  }
  return Logits(directory, out_path, tokens);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 1) {
    std::cout << "ember " << ember::version() << '\n';
    return 0;
  }
  const std::string_view command = argv[1];
  if (command == "inspect" && argc == 3) return Inspect(argv[2]);
  if (command == "logits") return RunLogits(argc, argv);
  return Usage();
}
