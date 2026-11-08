// libFuzzer entry point for the JSON parser and the config reader. Any input must parse or fail,
// never crash, read out of bounds, or overflow the stack. Build with the fuzz preset; see
// docs/loader.md.

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "ember/json.h"
#include "ember/llama.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::string_view text(reinterpret_cast<const char*>(data), size);
  const auto parsed = ember::Json::Parse(text);
  if (parsed.ok() && parsed->is_object()) {
    // Walk what came out, so a parse that built something broken is caught too.
    for (std::size_t i = 0; i < parsed->size(); ++i) {
      static_cast<void>(parsed->Find(parsed->keys()[i]));
    }
  }
  // The same text as a model config, which is where JSON values turn into sizes the engine trusts.
  static_cast<void>(ember::LlamaConfig::FromJson(text));
  return 0;
}
