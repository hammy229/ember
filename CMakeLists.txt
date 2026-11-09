cmake_minimum_required(VERSION 3.24)
project(ember VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(EMBER_BUILD_TESTS "Build the test suite" ${PROJECT_IS_TOP_LEVEL})
option(EMBER_WERROR "Treat warnings as errors" ${PROJECT_IS_TOP_LEVEL})
option(EMBER_SANITIZE "Build with AddressSanitizer and UBSan" OFF)
option(EMBER_FUZZ "Build the libFuzzer targets in fuzz/ (needs a clang with libFuzzer)" OFF)

if(EMBER_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
endif()

if(EMBER_FUZZ)
  # Everything gets coverage hooks so the fuzzer can see which branches an input reached. Only the
  # fuzz targets link libFuzzer's main().
  add_compile_options(-fsanitize=fuzzer-no-link,address,undefined -fno-omit-frame-pointer
                      -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
endif()

# One place for the warning flags. Every target links this.
add_library(ember_warnings INTERFACE)
target_compile_options(ember_warnings INTERFACE -Wall -Wextra -Wpedantic -Wconversion)
if(EMBER_WERROR)
  target_compile_options(ember_warnings INTERFACE -Werror)
endif()

configure_file(src/version.cpp.in generated/version.cpp @ONLY)

add_library(ember STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/generated/version.cpp
  src/aligned_buffer.cpp
  src/arena.cpp
  src/check.cpp
  src/f16.cpp
  src/forward.cpp
  src/json.cpp
  src/layers.cpp
  src/llama.cpp
  src/mapped_file.cpp
  src/ops.cpp
  src/safetensors.cpp
  src/shape.cpp
  src/status.cpp
  src/tensor.cpp)
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
    -sEXPORTED_FUNCTIONS=_ember_version,_ember_memory_self_check,_ember_ops_self_check,_ember_attention_frame,_ember_softmax_row,_ember_matmul_bench,_malloc,_free
    -sEXPORTED_RUNTIME_METHODS=UTF8ToString,HEAPF32
    --no-entry)
  return()
endif()

if(EMBER_FUZZ)
  foreach(target json safetensors)
    add_executable(ember_fuzz_${target} fuzz/${target}_fuzz.cpp)
    target_link_libraries(ember_fuzz_${target} PRIVATE ember ember_warnings)
    target_link_options(ember_fuzz_${target} PRIVATE -fsanitize=fuzzer)
  endforeach()
  add_executable(ember_fuzz_seeds fuzz/make_seeds.cpp)
  target_link_libraries(ember_fuzz_seeds PRIVATE ember_warnings)
endif()

add_executable(ember_cli tools/ember_cli.cpp)
target_link_libraries(ember_cli PRIVATE ember)
target_link_libraries(ember_cli PRIVATE ember_warnings)
set_target_properties(ember_cli PROPERTIES OUTPUT_NAME "ember")

add_executable(ember_bench tools/ember_bench.cpp)
target_link_libraries(ember_bench PRIVATE ember ember_warnings)

add_executable(ember_bench_ops tools/ember_bench_ops.cpp)
target_link_libraries(ember_bench_ops PRIVATE ember ember_warnings)

add_executable(ember_bench_model tools/ember_bench_model.cpp)
target_link_libraries(ember_bench_model PRIVATE ember ember_warnings)

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
  add_executable(ember_tests
    tests/aligned_buffer_test.cpp
    tests/arena_random_test.cpp
    tests/arena_test.cpp
    tests/check_test.cpp
    tests/dtype_test.cpp
    tests/f16_test.cpp
    tests/forward_test.cpp
    tests/hf_reference_test.cpp
    tests/json_test.cpp
    tests/layers_test.cpp
    tests/llama_load_test.cpp
    tests/loader_corruption_test.cpp
    tests/mapped_file_test.cpp
    tests/ops_test.cpp
    tests/safetensors_test.cpp
    tests/shape_test.cpp
    tests/status_or_test.cpp
    tests/status_test.cpp
    tests/tensor_test.cpp
    tests/timer_test.cpp
    tests/version_test.cpp)
  target_link_libraries(ember_tests PRIVATE ember GTest::gtest_main)
  target_link_libraries(ember_tests PRIVATE ember_warnings)
  target_compile_definitions(ember_tests PRIVATE EMBER_PROJECT_VERSION="${PROJECT_VERSION}"
    EMBER_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/reference")

  include(GoogleTest)
  gtest_discover_tests(ember_tests)

  # The executables get run too, so a broken main() fails the suite.
  add_test(NAME cli_prints_version COMMAND ember_cli)
  set_tests_properties(cli_prints_version PROPERTIES
    PASS_REGULAR_EXPRESSION "^ember ${PROJECT_VERSION}\n$")

  add_test(NAME cli_rejects_unknown_command COMMAND ember_cli bogus)
  set_tests_properties(cli_rejects_unknown_command PROPERTIES PASS_REGULAR_EXPRESSION "usage:")
  add_test(NAME cli_inspect_reports_a_missing_model COMMAND ember_cli inspect /no/such/model)
  set_tests_properties(cli_inspect_reports_a_missing_model PROPERTIES
    PASS_REGULAR_EXPRESSION "NOT_FOUND")

  add_test(NAME cli_logits_rejects_a_bad_token_id COMMAND ember_cli logits /no/model 1 x)
  set_tests_properties(cli_logits_rejects_a_bad_token_id PROPERTIES PASS_REGULAR_EXPRESSION "usage:")

  add_test(NAME bench_model_rejects_bad_repeats COMMAND ember_bench_model /no/model 0)
  set_tests_properties(bench_model_rejects_bad_repeats PROPERTIES PASS_REGULAR_EXPRESSION "usage:")

  add_test(NAME bench_runs COMMAND ember_bench 1 1000)
  set_tests_properties(bench_runs PROPERTIES PASS_REGULAR_EXPRESSION "median")

  foreach(bad_args "abc" "0" "-1" "5x" "1 0" "1 2 3" "99999999999999999999" "1 2000000000"
          "1000001" "18446744073709551615")
    string(REPLACE " " "_" name "${bad_args}")
    separate_arguments(args UNIX_COMMAND "${bad_args}")
    add_test(NAME bench_rejects_${name} COMMAND ember_bench ${args})
    set_tests_properties(bench_rejects_${name} PROPERTIES PASS_REGULAR_EXPRESSION "usage:")
  endforeach()

  add_test(NAME bench_ops_runs COMMAND ember_bench_ops 1 8)
  set_tests_properties(bench_ops_runs PROPERTIES PASS_REGULAR_EXPRESSION "softmax 8x8, each column")

  foreach(bad_args "abc" "0" "1 0" "1 2 3" "1 4097" "100001")
    string(REPLACE " " "_" name "${bad_args}")
    separate_arguments(args UNIX_COMMAND "${bad_args}")
    add_test(NAME bench_ops_rejects_${name} COMMAND ember_bench_ops ${args})
    set_tests_properties(bench_ops_rejects_${name} PROPERTIES PASS_REGULAR_EXPRESSION "usage:")
  endforeach()
endif()
