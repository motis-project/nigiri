cmake_minimum_required(VERSION 3.22)
project(nigiri)

include(cmake/buildcache.cmake)
include(cmake/pkg.cmake)

# --- LIB ---
file(GLOB_RECURSE nigiri-files src/*.cc)
add_library(nigiri ${nigiri-files})
target_include_directories(
  nigiri
  PUBLIC include/public
  PUBLIC include/private
)
target_link_libraries(nigiri PUBLIC cista geo utl lmdb fmt date miniz date-tz)
target_compile_features(nigiri PUBLIC cxx_std_20)

# --- TEST ---
add_library(nigiri-generated INTERFACE)
target_include_directories(nigiri-generated INTERFACE ${CMAKE_BINARY_DIR}/generated)

configure_file (
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_dir.h.in
  ${CMAKE_BINARY_DIR}/generated/test_dir.h
)
file(GLOB_RECURSE nigiri-test-files test/*.cc)
add_executable(nigiri-test ${nigiri-test-files})
target_link_libraries(nigiri-test nigiri doctest ianatzdb-res nigiri-generated)
