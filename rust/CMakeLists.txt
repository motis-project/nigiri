corrosion_import_crate(MANIFEST_PATH Cargo.toml)
corrosion_add_cxxbridge(
  nigiri-cxx
  CRATE srv
  FILES main.rs
)
target_compile_features(nigiri-cxx PRIVATE cxx_std_23)
target_sources(nigiri-cxx PRIVATE cpp/src/rust.cc)
target_link_libraries(nigiri-cxx nigiri)
target_include_directories(nigiri-cxx PRIVATE "cpp/include")
corrosion_link_libraries(srv nigiri-cxx nigiri date-tz ianatzdb geo utl fmt miniz gtfsrt)