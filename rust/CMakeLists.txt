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
add_library(tz-res
    ${CMAKE_BINARY_DIR}/deps/date/ianatzdb/obj/res_0.o
    ${CMAKE_BINARY_DIR}/deps/date/ianatzdb/obj/res_1.o
    ${CMAKE_BINARY_DIR}/deps/date/ianatzdb/obj/res_2.o
    ${CMAKE_BINARY_DIR}/deps/date/ianatzdb/obj/res_3.o
    ${CMAKE_BINARY_DIR}/deps/date/ianatzdb/obj/res_4.o
    ${CMAKE_BINARY_DIR}/deps/date/ianatzdb/obj/res_5.o
    ${CMAKE_BINARY_DIR}/deps/date/ianatzdb/obj/res_6.o
    ${CMAKE_BINARY_DIR}/deps/date/ianatzdb/obj/res_7.o
    ${CMAKE_BINARY_DIR}/deps/date/ianatzdb/obj/res_8.o
    ${CMAKE_BINARY_DIR}/deps/date/ianatzdb/obj/res_9.o
    ${CMAKE_BINARY_DIR}/deps/date/ianatzdb/obj/res_10.o
    ${CMAKE_BINARY_DIR}/deps/date/ianatzdb/obj/res_11.o
    ${CMAKE_BINARY_DIR}/deps/date/ianatzdb/obj/res_12.o
)
set_target_properties(tz-res PROPERTIES LINKER_LANGUAGE CXX)
corrosion_link_libraries(srv nigiri-cxx nigiri date-tz ianatzdb tz-res geo utl fmt miniz gtfsrt)