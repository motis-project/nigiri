# Redirects minizip-ng's find_package(zstd QUIET CONFIG) to the zstd vendored
# in .pkg (not built by deps/CMakeLists.txt, since upstream zstd has no root
# CMakeLists.txt - its CMake project lives in build/cmake/), using the same
# binary directory the deps tree would use.

get_filename_component(_nigiri_zstd_root "../../../../deps/zstd" ABSOLUTE
      BASE_DIR ${CMAKE_CURRENT_LIST_DIR})
set(zstd_VERSION 1.5.7)
set(zstd_FOUND ON)

# zstd only defines these aliases inside its export sets, so create the ones
# minizip-ng links against (BUILD_SHARED_LIBS is OFF here) in this scope,
# where minizip-ng's targets are defined.
add_library(zstd::libzstd_static ALIAS libzstd_static)
add_library(zstd::libzstd_shared ALIAS libzstd_shared)
