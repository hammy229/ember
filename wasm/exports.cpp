#include <emscripten/emscripten.h>

#include "ember/version.h"

extern "C" {

EMSCRIPTEN_KEEPALIVE const char* ember_version() { return ember::version(); }

}  // extern "C"
