# Shim package-config files that redirect minizip-ng's find_package() calls to
# the pinned in-tree dependency builds. They are found via CMAKE_PREFIX_PATH:
# layout "cmake/minizip-ng-deps/<name>/cmake/<name>Config.cmake" matches one of
# the standard config-search patterns for prefix <dirname>.
#
# minizip-ng is added last by pkg (alphabetically after zstd among the new
# deps), so zstd is not yet - and will never be - add_subdirectory'ed by
# deps/CMakeLists.txt: upstream zstd has no root CMakeLists.txt, its CMake
# project lives in build/cmake/. Hence the zstd shim config configures
# deps/zstd/build/cmake in place when minizip-ng calls
# find_package(zstd QUIET CONFIG).

set(_minizip_deps_prefix ${CMAKE_CURRENT_LIST_DIR}/minizip-ng-deps)
list(APPEND CMAKE_PREFIX_PATH ${_minizip_deps_prefix})
list(APPEND CMAKE_MODULE_PATH ${_minizip_deps_prefix}/zlib-modules)
