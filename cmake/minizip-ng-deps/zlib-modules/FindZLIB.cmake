# Module-mode shim: resolves find_package(ZLIB) to the zlib-ng

set(ZLIB_FOUND TRUE)
set(ZLIB_VERSION_STRING "1.3.1")
set(ZLIB_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/deps/zlib-ng)

add_library(ZLIB::ZLIB INTERFACE IMPORTED)
