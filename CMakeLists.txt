cmake_minimum_required(VERSION 3.22)
project(nigiri)

include(cmake/pkg.cmake)

# --- LIB ---
file(GLOB_RECURSE nigiri-files src/*.cc)
add_library(nigiri ${nigiri-files})
target_include_directories(
  nigiri
  PUBLIC include/public
  PUBLIC include/private
)
target_link_libraries(nigiri PUBLIC cista geo utl lmdb fmt date)
target_compile_features(nigiri PUBLIC cxx_std_20)

# --- TEST ---
file(GLOB_RECURSE nigiri-test-files test/*.cc)
add_executable(nigiri-test ${nigiri-test-files})
target_link_libraries(nigiri-test nigiri doctest ianatzdb-res)
