cmake_minimum_required(VERSION 3.24)
project(ember VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(EMBER_BUILD_TESTS "Build the test suite" ON)
option(EMBER_SANITIZE "Build with AddressSanitizer and UBSan" OFF)

if(EMBER_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
endif()

# One place for the warning flags. Every target links this.
add_library(ember_warnings INTERFACE)
target_compile_options(ember_warnings INTERFACE -Wall -Wextra -Wpedantic -Wconversion -Werror)

configure_file(src/version.cpp.in generated/version.cpp @ONLY)

add_library(ember STATIC ${CMAKE_CURRENT_BINARY_DIR}/generated/version.cpp)
target_include_directories(ember PUBLIC include)
target_link_libraries(ember PRIVATE ember_warnings)

if(EMSCRIPTEN)
  add_executable(ember_wasm wasm/exports.cpp)
  target_link_libraries(ember_wasm PRIVATE ember)
  target_link_libraries(ember_wasm PRIVATE ember_warnings)
  set_target_properties(ember_wasm PROPERTIES OUTPUT_NAME "ember" SUFFIX ".js")
  target_link_options(ember_wasm PRIVATE
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sENVIRONMENT=web,node
    -sALLOW_MEMORY_GROWTH=1
    -sEXPORTED_FUNCTIONS=_ember_version
    -sEXPORTED_RUNTIME_METHODS=UTF8ToString
    --no-entry)
  return()
endif()

add_executable(ember_cli tools/ember_cli.cpp)
target_link_libraries(ember_cli PRIVATE ember)
target_link_libraries(ember_cli PRIVATE ember_warnings)
set_target_properties(ember_cli PROPERTIES OUTPUT_NAME "ember")

if(EMBER_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(ember_tests tests/version_test.cpp)
  target_link_libraries(ember_tests PRIVATE ember GTest::gtest_main)
  target_link_libraries(ember_tests PRIVATE ember_warnings)

  include(GoogleTest)
  gtest_discover_tests(ember_tests)
endif()
