#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
#include "ember/json.h"
#include "ember/llama.h"
#include "ember/route.h"
#include "ember/threads.h"
#include "ember/timer.h"
#include "ember/tokenizer.h"
#include "ember/version.h"

namespace {

int Usage() {
  std::cerr
      << "usage: ember                    print the version\n"
         "       ember inspect MODEL_DIR  load a model and describe it\n"
         "       ember logits MODEL_DIR [--out FILE] [--q8] [--threads N] TOKEN_ID...\n"
         "                                run the model and show the likeliest next tokens;\n"
         "                                --out saves every logit as little-endian f32\n"
         "       ember tokenize MODEL_DIR TEXT  print the token ids of TEXT, one line\n"
         "       ember route MODEL_DIR --threshold T [--signal NAME] [--tokens N] [--threads N]\n"
         "                   [--q8] PROMPT\n"
         "                                answer locally in chat format, and stop the moment the\n"
         "                                model gets unsure enough that a bigger model should "
         "take\n"
         "                                over; NAME is \"max entropy\" (default), \"lowest "
         "margin\",\n"
         "                                \"mean entropy\", or \"share of tokens with margin under "
         "0.5\"\n"
         "       ember generate MODEL_DIR [options] PROMPT\n"
         "                                continue the prompt, printing text as it comes\n"
         "         --tokens N   new tokens at most (default 128)\n"
         "         --temp T     temperature, 0 for greedy (default 0)\n"
         "         --top-k K    keep the K likeliest tokens (default all)\n"
         "         --top-p P    keep the likeliest tokens up to probability P (default 1)\n"
         "         --seed S     random seed (default 0)\n"
         "         --stats      show each token's entropy and top-2 margin afterwards\n"
         "         --threads N  threads for the math (default 1)\n"
         "         --q8         store the matrices in 8 bits (smaller and faster)\n"
         "         --chat       wrap the prompt as one user turn in the model's chat format\n"
         "                      (Qwen3, with thinking turned off)\n"
         "       ember batch MODEL_DIR IN.jsonl OUT.jsonl [--tokens N] [--threads N] [--chat] "
         "[--q8]\n"
         "                                answer each {\"id\", \"prompt\"} line greedily, writing\n"
         "                                the text and every token's entropy and margin\n";
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
           const std::vector<std::int32_t>& tokens, const ember::LoadOptions& load) {
  const auto model = ember::LlamaModel::Load(directory, load);
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
  ember::LoadOptions load;
  std::vector<std::int32_t> tokens;
  for (int i = 3; i < argc; ++i) {
    const std::string_view text = argv[i];
    if (text == "--out" && i + 1 < argc) {
      out_path = argv[++i];
      continue;
    }
    if (text == "--q8") {
      load.q8 = true;
      continue;
    }
    if (text == "--threads" && i + 1 < argc) {
      int threads = 0;
      const std::string_view value = argv[++i];
      const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), threads);
      if (error != std::errc() || end != value.data() + value.size() || threads < 1) return Usage();
      ember::SetThreadCount(static_cast<std::size_t>(threads));
      continue;
    }
    std::int32_t token = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), token);
    if (error != std::errc() || end != text.data() + text.size()) return Usage();
    tokens.push_back(token);
  }
  if (tokens.empty()) return Usage();
  return Logits(directory, out_path, tokens, load);
}

int RunTokenize(int argc, char** argv) {
  if (argc != 4) return Usage();
  const auto tokenizer = ember::Tokenizer::Load(std::string(argv[2]) + "/tokenizer.json");
  if (!tokenizer.ok()) {
    std::cerr << "error: " << tokenizer.status().ToString() << '\n';
    return 1;
  }
  const std::vector<std::int32_t> ids = tokenizer->Encode(argv[3], true);
  for (std::size_t i = 0; i < ids.size(); ++i) std::cout << (i ? " " : "") << ids[i];
  std::cout << '\n';
  return 0;
}

// Parses a whole argument as a number, or fails.
template <typename T>
bool ParseNumber(std::string_view text, T& value) {
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc() && end == text.data() + text.size();
}

// Text as a JSON string, quotes included.
std::string JsonString(std::string_view text) {
  std::string out = "\"";
  for (const char c : text) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char escaped[8];
          std::snprintf(escaped, sizeof escaped, "\\u%04x", static_cast<unsigned>(c));
          out += escaped;
        } else {
          out += c;
        }
    }
  }
  return out + "\"";
}

// One user turn in Qwen3's chat format, with thinking turned off.
std::string ChatPrompt(std::string_view prompt) {
  return "<|im_start|>user\n" + std::string(prompt) +
         "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

// Loads the model once and answers many prompts, for evaluations like scripts/route_eval.py.
int RunBatch(int argc, char** argv) {
  if (argc < 5) return Usage();
  const std::string directory = argv[2];
  const std::string in_path = argv[3];
  const std::string out_path = argv[4];
  ember::GenerateOptions options;
  options.max_new_tokens = 256;
  bool chat = false;
  ember::LoadOptions load;
  for (int i = 5; i < argc; ++i) {
    const std::string_view arg = argv[i];
    bool ok = true;
    if (arg == "--tokens" && i + 1 < argc) {
      ok = ParseNumber(argv[++i], options.max_new_tokens);
    } else if (arg == "--threads" && i + 1 < argc) {
      std::size_t threads = 0;
      ok = ParseNumber(argv[++i], threads) && threads >= 1 && threads <= 256;
      if (ok) ember::SetThreadCount(threads);
    } else if (arg == "--chat") {
      chat = true;
    } else if (arg == "--q8") {
      load.q8 = true;
    } else {
      ok = false;
    }
    if (!ok) return Usage();
  }
  const auto model = ember::LlamaModel::Load(directory, load);
  if (!model.ok()) {
    std::cerr << "error: " << model.status().ToString() << '\n';
    return 1;
  }
  const auto tokenizer = ember::Tokenizer::Load(directory + "/tokenizer.json");
  if (!tokenizer.ok()) {
    std::cerr << "error: " << tokenizer.status().ToString() << '\n';
    return 1;
  }
  if (chat && tokenizer->kind() != ember::Tokenizer::Kind::kByteLevel) {
    std::cerr << "error: --chat needs a model with a chat format, like Qwen3\n";
    return 1;
  }
  options.stop_tokens = tokenizer->stop_ids();
  std::ifstream in(in_path);
  std::ofstream out(out_path, std::ios::trunc);
  if (!in || !out) {
    std::cerr << "error: could not open " << (in ? out_path : in_path) << '\n';
    return 1;
  }
  std::string line;
  std::size_t done = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const auto record = ember::Json::Parse(line);
    const bool object = record.ok() && record->is_object();
    const ember::Json* id = object ? record->Find("id") : nullptr;
    const ember::Json* prompt = object ? record->Find("prompt") : nullptr;
    if (id == nullptr || prompt == nullptr || !prompt->is_string() ||
        !(id->is_string() || id->as_uint64().has_value())) {
      std::cerr << "error: each line needs an \"id\" and a string \"prompt\"\n";
      return 1;
    }
    const std::string text = chat ? ChatPrompt(prompt->as_string()) : prompt->as_string();
    const std::vector<std::int32_t> ids = tokenizer->Encode(text, true);
    const ember::Stopwatch stopwatch;
    const auto result = ember::Generate(**model, ids, options);
    const double ms = stopwatch.ElapsedMs();
    if (!result.ok()) {
      std::cerr << "error: " << result.status().ToString() << '\n';
      return 1;
    }
    std::vector<std::int32_t> reply;
    for (const auto& t : result->tokens) {
      if (std::find(options.stop_tokens.begin(), options.stop_tokens.end(), t.id) ==
          options.stop_tokens.end()) {
        reply.push_back(t.id);
      }
    }
    const std::string id_json =
        id->is_string() ? JsonString(id->as_string()) : std::to_string(*id->as_uint64());
    out << "{\"id\": " << id_json << ", \"text\": " << JsonString(tokenizer->Decode(reply).value())
        << ", \"prompt_tokens\": " << ids.size() << ", \"ms\": " << ms
        << ", \"stop\": " << static_cast<int>(result->stop) << ", \"tokens\": [";
    for (std::size_t i = 0; i < result->tokens.size(); ++i) {
      const auto& t = result->tokens[i];
      out << (i ? ", " : "") << "[" << t.id << ", " << t.stats.entropy << ", " << t.stats.margin
          << ", " << t.stats.top1_prob << "]";
    }
    out << "]}\n" << std::flush;
    ++done;
    std::cerr << "\r" << done << " done" << std::flush;
  }
  std::cerr << '\n';
  return 0;
}

// Answers locally and stops as soon as the router says the request should go to a bigger model.
int RunRoute(int argc, char** argv) {
  if (argc < 4) return Usage();
  const std::string directory = argv[2];
  ember::GenerateOptions options;
  options.max_new_tokens = 384;
  ember::LoadOptions load;
  ember::Router::Options policy;
  bool have_threshold = false;
  std::string prompt;
  for (int i = 3; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const bool has_value = i + 1 < argc;
    bool ok = true;
    if (arg == "--threshold" && has_value) {
      ok = ParseNumber(argv[++i], policy.threshold);
      have_threshold = ok;
    } else if (arg == "--signal" && has_value) {
      const auto signal = ember::Router::ParseSignal(argv[++i]);
      ok = signal.ok();
      if (ok) policy.signal = *signal;
    } else if (arg == "--tokens" && has_value) {
      ok = ParseNumber(argv[++i], options.max_new_tokens);
    } else if (arg == "--threads" && has_value) {
      std::size_t threads = 0;
      ok = ParseNumber(argv[++i], threads) && threads >= 1 && threads <= 256;
      if (ok) ember::SetThreadCount(threads);
    } else if (arg == "--q8") {
      load.q8 = true;
    } else if (i == argc - 1 && !arg.starts_with("--")) {
      prompt = arg;
    } else {
      ok = false;
    }
    if (!ok) return Usage();
  }
  if (prompt.empty() || !have_threshold) return Usage();
  const auto model = ember::LlamaModel::Load(directory, load);
  if (!model.ok()) {
    std::cerr << "error: " << model.status().ToString() << '\n';
    return 1;
  }
  const auto tokenizer = ember::Tokenizer::Load(directory + "/tokenizer.json");
  if (!tokenizer.ok()) {
    std::cerr << "error: " << tokenizer.status().ToString() << '\n';
    return 1;
  }
  if (tokenizer->kind() != ember::Tokenizer::Kind::kByteLevel) {
    std::cerr << "error: ember route needs a chat model, like Qwen3\n";
    return 1;
  }
  options.stop_tokens = tokenizer->stop_ids();
  const std::vector<std::int32_t> ids = tokenizer->Encode(ChatPrompt(prompt), true);
  auto generator = ember::Generator::Make(**model, ids, options);
  if (!generator.ok()) {
    std::cerr << "error: " << generator.status().ToString() << '\n';
    return 1;
  }
  ember::Router router(policy);
  std::vector<std::int32_t> reply;
  std::size_t printed = 0;
  const ember::Stopwatch stopwatch;
  while (true) {
    const auto token = (*generator)->Next();
    if (!token.ok()) {
      std::cerr << "\nerror: " << token.status().ToString() << '\n';
      return 1;
    }
    if (!token->has_value()) break;
    const ember::GeneratedToken& t = **token;
    if (router.Update(t.stats)) break;
    if (std::find(options.stop_tokens.begin(), options.stop_tokens.end(), t.id) !=
        options.stop_tokens.end()) {
      continue;
    }
    reply.push_back(t.id);
    const std::string text = tokenizer->Decode(reply).value();
    if (text.size() > printed && !text.ends_with("\xEF\xBF\xBD")) {
      std::cout << text.substr(printed) << std::flush;
      printed = text.size();
    }
  }
  const double ms = stopwatch.ElapsedMs();
  std::cout << '\n';
  if (router.escalated()) {
    std::cout << "[escalated at token " << *router.escalated_at() << ": score " << router.score()
              << " reached the threshold " << policy.threshold
              << ". A local-first router would hand this request to a bigger model now.]\n";
  } else {
    std::cout << "[answered locally: the score stayed at " << router.score() << ", under "
              << policy.threshold << ".]\n";
  }
  std::cerr << router.tokens() << " tokens in " << ms << " ms\n";
  return 0;
}

int RunGenerate(int argc, char** argv) {
  if (argc < 4) return Usage();
  const std::string directory = argv[2];
  ember::GenerateOptions options;
  options.max_new_tokens = 128;
  bool show_stats = false;
  bool chat = false;
  ember::LoadOptions load;
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
    } else if (arg == "--chat") {
      chat = true;
    } else if (arg == "--q8") {
      load.q8 = true;
    } else if (arg == "--threads" && has_value) {
      std::size_t threads = 0;
      ok = ParseNumber(argv[++i], threads) && threads >= 1 && threads <= 256;
      if (ok) ember::SetThreadCount(threads);
    } else if (i == argc - 1 && !arg.starts_with("--")) {
      prompt = arg;
    } else {
      ok = false;
    }
    if (!ok) return Usage();
  }
  if (prompt.empty()) return Usage();

  const auto model = ember::LlamaModel::Load(directory, load);
  if (!model.ok()) {
    std::cerr << "error: " << model.status().ToString() << '\n';
    return 1;
  }
  const auto tokenizer = ember::Tokenizer::Load(directory + "/tokenizer.json");
  if (!tokenizer.ok()) {
    std::cerr << "error: " << tokenizer.status().ToString() << '\n';
    return 1;
  }
  if (chat) {
    if (tokenizer->kind() != ember::Tokenizer::Kind::kByteLevel) {
      std::cerr << "error: --chat needs a model with a chat format, like Qwen3\n";
      return 1;
    }
    prompt = ChatPrompt(prompt);
  }
  const std::vector<std::int32_t> prompt_ids = tokenizer->Encode(prompt, true);
  options.stop_tokens = tokenizer->stop_ids();
  const auto is_stop = [&](std::int32_t id) {
    return std::find(options.stop_tokens.begin(), options.stop_tokens.end(), id) !=
           options.stop_tokens.end();
  };

  // Decoding the prompt and the new tokens together keeps the space before the first new word.
  // Only text that is settled gets printed: a run of byte pieces still waiting for the rest of its
  // character decodes to U+FFFD for now.
  // A chat prompt is not echoed, only the reply.
  const bool skip_bos = !prompt_ids.empty() && prompt_ids[0] == tokenizer->bos_id();
  std::vector<std::int32_t> text_ids(prompt_ids.begin() + (skip_bos ? 1 : 0), prompt_ids.end());
  if (chat) text_ids.clear();
  const std::string prompt_text = tokenizer->Decode(text_ids).value();
  std::size_t printed = prompt_text.size();
  std::cout << prompt_text << std::flush;
  const auto on_token = [&](const ember::GeneratedToken& token, std::size_t) {
    if (is_stop(token.id)) return true;
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
  if (command == "batch") return RunBatch(argc, argv);
  if (command == "tokenize") return RunTokenize(argc, argv);
  if (command == "route") return RunRoute(argc, argv);
  return Usage();
}
