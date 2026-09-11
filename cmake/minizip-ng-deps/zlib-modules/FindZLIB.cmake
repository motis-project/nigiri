# Module-mode shim: resolves find_package(ZLIB) to the zlib-ng already built
# in the tree (deps/zlib-ng, ZLIB_COMPAT=ON: standard zlib API/headers, target
# zlibstatic). minizip-ng is configured after zlib-ng but its other flavors
# don't fit: with MZ_ZLIB_FLAVOR=zlib-ng it expects the native zlib-ng API
# (zlib-ng.h), while our in-tree zlib-ng is built with ZLIB_COMPAT=ON.
# No other dependency in deps/ runs find_package(ZLIB) (curl only checks for
# an already-existing ZLIB::ZLIB target), so this hijack has no side effects.

set(ZLIB_FOUND TRUE)
set(ZLIB_VERSION_STRING "1.3.1")
set(ZLIB_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/deps/zlib-ng)

# Legacy result variables for consumers reading ZLIB_LIBRARIES directly.
set(ZLIB_LIBRARIES ZLIB::ZLIB)

add_library(ZLIB::ZLIB INTERFACE IMPORTED)
set_target_properties(ZLIB::ZLIB PROPERTIES
      INTERFACE_LINK_LIBRARIES zlibstatic
      INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR}/deps/zlib-ng)
