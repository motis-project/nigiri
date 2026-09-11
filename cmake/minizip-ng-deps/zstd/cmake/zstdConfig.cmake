# Redirects minizip-ng's find_package(zstd QUIET CONFIG) to the zstd vendored
# in .pkg (not built by deps/CMakeLists.txt, since upstream zstd has no root
# CMakeLists.txt - its CMake project lives in build/cmake/), using the same
# binary directory the deps tree would use.

get_filename_component(_nigiri_zstd_root "../../../../deps/zstd" ABSOLUTE
      BASE_DIR ${CMAKE_CURRENT_LIST_DIR})
set(zstd_VERSION 1.5.7)
set(zstd_FOUND ON)

set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "Build zstd programs" FORCE)
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "Build zstd tests" FORCE)

add_subdirectory(${_nigiri_zstd_root}/build/cmake
      ${CMAKE_BINARY_DIR}/deps/zstd EXCLUDE_FROM_ALL)

# zstd only defines these aliases inside its export sets, so create the ones
# minizip-ng links against (BUILD_SHARED_LIBS is OFF here) in this scope,
# where minizip-ng's targets are defined.
add_library(zstd::libzstd_static ALIAS libzstd_static)
add_library(zstd::libzstd_shared ALIAS libzstd_shared)
