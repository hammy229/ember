#include <cstddef>
#include <iostream>
#include <string>
#include <string_view>

#include "ember/llama.h"
#include "ember/timer.h"
#include "ember/version.h"

namespace {

int Usage() {
  std::cerr << "usage: ember                    print the version\n"
               "       ember inspect MODEL_DIR  load a model and describe it\n";
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

}  // namespace

int main(int argc, char** argv) {
  if (argc == 1) {
    std::cout << "ember " << ember::version() << '\n';
    return 0;
  }
  const std::string_view command = argv[1];
  if (command == "inspect" && argc == 3) return Inspect(argv[2]);
  return Usage();
}
