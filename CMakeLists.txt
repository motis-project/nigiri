project(nigiri)
cmake_minimum_required(VERSION 3.23)

include(cmake/pkg.cmake)

# --- LIB ---
file(GLOB_RECURSE nigiri-files src/*.cc)
add_library(nigiri ${nigiri-files})
target_include_directories(
  nigiri
  PUBLIC include/public
  PRIVATE include/private
)
target_link_libraries(nigiri PUBLIC cista geo)
target_compile_features(nigiri PUBLIC cxx_std_20)

# --- TEST ---
file(GLOB_RECURSE nigiri-test-files test/*.cc)
add_executable(nigiri-test ${nigiri-test-files})
target_link_libraries(nigiri-test nigiri doctest)
