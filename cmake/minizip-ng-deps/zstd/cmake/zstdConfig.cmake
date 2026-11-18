# zstd only defines these aliases inside its export sets, so create the ones
# minizip-ng links against
add_library(zstd::libzstd_static ALIAS libzstd_static)
