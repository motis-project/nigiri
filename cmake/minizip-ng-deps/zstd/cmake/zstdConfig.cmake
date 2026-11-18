# Redirects minizip-ng's find_package(zstd QUIET CONFIG) to the zstd vendored
# in .pkg

get_filename_component(_nigiri_zstd_root "../../../../deps/zstd" ABSOLUTE
      BASE_DIR ${CMAKE_CURRENT_LIST_DIR})
set(zstd_VERSION 1.5.7)
set(zstd_FOUND ON)

# zstd only defines these aliases inside its export sets, so create the ones
# minizip-ng links against
add_library(zstd::libzstd_static ALIAS libzstd_static)
