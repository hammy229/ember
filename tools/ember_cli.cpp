#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "ember/forward.h"
#include "ember/generate.h"
#include "ember/llama.h"
#include "ember/timer.h"
#include "ember/tokenizer.h"
#include "ember/version.h"

namespace {

int Usage() {
  std::cerr << "usage: ember                    print the version\n"
               "       ember inspect MODEL_DIR  load a model and describe it\n"
               "       ember logits MODEL_DIR [--out FILE] TOKEN_ID...\n"
               "                                run the model and show the likeliest next tokens;\n"
               "                                --out saves every logit as little-endian f32\n"
               "       ember generate MODEL_DIR [options] PROMPT\n"
               "                                continue the prompt, printing text as it comes\n"
               "         --tokens N   new tokens at most (default 128)\n"
               "         --temp T     temperature, 0 for greedy (default 0)\n"
               "         --top-k K    keep the K likeliest tokens (default all)\n"
               "         --top-p P    keep the likeliest tokens up to probability P (default 1)\n"
               "         --seed S     random seed (default 0)\n"
               "         --stats      show each token's entropy and top-2 margin afterwards\n";
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
            << "widened to f32 " << (*model)->widened_bytes() << " bytes\n"
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

// Parses a whole argument as a number, or fails.
template <typename T>
bool ParseNumber(std::string_view text, T& value) {
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc() && end == text.data() + text.size();
}

int RunGenerate(int argc, char** argv) {
  if (argc < 4) return Usage();
  const std::string directory = argv[2];
  ember::GenerateOptions options;
  options.max_new_tokens = 128;
  bool show_stats = false;
  std::string prompt;
  for (int i = 3; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const bool has_value = i + 1 < argc;
    bool ok = true;
    if (arg == "--tokens" && has_value) {
      ok = ParseNumber(argv[++i], options.max_new_tokens);
    } else if (arg == "--temp" && has_value) {
      ok = ParseNumber(argv[++i], options.sampling.temperature);
    } else if (arg == "--top-k" && has_value) {
      ok = ParseNumber(argv[++i], options.sampling.top_k);
    } else if (arg == "--top-p" && has_value) {
      ok = ParseNumber(argv[++i], options.sampling.top_p);
    } else if (arg == "--seed" && has_value) {
      ok = ParseNumber(argv[++i], options.sampling.seed);
    } else if (arg == "--stats") {
      show_stats = true;
    } else if (i == argc - 1 && !arg.starts_with("--")) {
      prompt = arg;
    } else {
      ok = false;
    }
    if (!ok) return Usage();
  }
  if (prompt.empty()) return Usage();

  const auto model = ember::LlamaModel::Load(directory);
  if (!model.ok()) {
    std::cerr << "error: " << model.status().ToString() << '\n';
    return 1;
  }
  const auto tokenizer = ember::Tokenizer::Load(directory + "/tokenizer.json");
  if (!tokenizer.ok()) {
    std::cerr << "error: " << tokenizer.status().ToString() << '\n';
    return 1;
  }
  const std::vector<std::int32_t> prompt_ids = tokenizer->Encode(prompt, true);
  options.stop_tokens = {tokenizer->eos_id(), tokenizer->bos_id()};

  // Decoding the prompt and the new tokens together keeps the space before the first new word.
  // Only text that is settled gets printed: a run of byte pieces still waiting for the rest of its
  // character decodes to U+FFFD for now.
  std::vector<std::int32_t> text_ids(prompt_ids.begin() + 1, prompt_ids.end());
  const std::string prompt_text = tokenizer->Decode(text_ids).value();
  std::size_t printed = prompt_text.size();
  std::cout << prompt_text << std::flush;
  const auto on_token = [&](const ember::GeneratedToken& token, std::size_t) {
    if (token.id == tokenizer->eos_id() || token.id == tokenizer->bos_id()) return true;
    text_ids.push_back(token.id);
    const auto text = tokenizer->Decode(text_ids);
    if (!text.ok() || text->size() < printed) return true;
    std::string_view fresh = std::string_view(*text).substr(printed);
    constexpr std::string_view kReplacement = "\xEF\xBF\xBD";
    if (fresh.ends_with(kReplacement)) return true;
    std::cout << fresh << std::flush;
    printed = text->size();
    return true;
  };

  const ember::Stopwatch stopwatch;
  const auto result = ember::Generate(**model, prompt_ids, options, on_token);
  const double ms = stopwatch.ElapsedMs();
  std::cout << '\n';
  if (!result.ok()) {
    std::cerr << "error: " << result.status().ToString() << '\n';
    return 1;
  }
  const std::size_t count = result->tokens.size();
  std::cerr << count << " tokens in " << ms << " ms ("
            << (ms > 0 ? 1000.0 * static_cast<double>(count) / ms : 0.0) << " tokens/s, prompt of "
            << prompt_ids.size() << " included)\n";
  if (show_stats) {
    std::cerr << std::fixed << std::setprecision(3) << "  #   entropy  margin  token\n";
    for (std::size_t i = 0; i < count; ++i) {
      const ember::GeneratedToken& t = result->tokens[i];
      std::cerr << std::setw(3) << i << "  " << std::setw(7) << t.stats.entropy << "  "
                << std::setw(6) << t.stats.margin << "  " << tokenizer->piece(t.id) << '\n';
    }
  }
  return 0;
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
  if (command == "generate") return RunGenerate(argc, argv);
  return Usage();
}
