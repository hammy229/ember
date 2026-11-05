#pragma once

// Lets the arena tell AddressSanitizer which of its bytes are really in use. In other builds these
// compile to nothing.

#if defined(__SANITIZE_ADDRESS__)
#define EMBER_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define EMBER_ASAN 1
#endif
#endif
#ifndef EMBER_ASAN
#define EMBER_ASAN 0
#endif

#if EMBER_ASAN
#include <sanitizer/asan_interface.h>
#define EMBER_POISON(address, bytes) __asan_poison_memory_region((address), (bytes))
#define EMBER_UNPOISON(address, bytes) __asan_unpoison_memory_region((address), (bytes))
#else
#define EMBER_POISON(address, bytes) ((void)(address), (void)(bytes))
#define EMBER_UNPOISON(address, bytes) ((void)(address), (void)(bytes))
#endif
