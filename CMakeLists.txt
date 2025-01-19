cmake_minimum_required(VERSION 3.22)
project(nigiri)

option(NIGIRI_CUDA "enable CUDA functions" OFF)
if (NIGIRI_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if (NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
  if (NIGIRI_MIMALLOC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  else ()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif ()
endif ()

include(cmake/buildcache.cmake)
include(cmake/pkg.cmake)


# --- GTFS-RT PROTOBUF ---
add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated/gtfsrt/gtfs-realtime.pb.h
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated/gtfsrt/gtfs-realtime.pb.cc
      COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} $<TARGET_FILE:protoc>
      --cpp_out=${CMAKE_CURRENT_BINARY_DIR}/generated/gtfsrt
      --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/gtfs-realtime.proto
      DEPENDS protoc
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gtfs-realtime.proto
      COMMENT "Generating GTFS RT protocol buffers C++ code"
)
add_library(gtfsrt ${CMAKE_CURRENT_BINARY_DIR}/generated/gtfsrt/gtfs-realtime.pb.cc)
target_compile_features(gtfsrt PUBLIC cxx_std_23)
target_include_directories(gtfsrt SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(gtfsrt protobuf::libprotobuf)


# --- LINT ---
option(NIGIRI_LINT "Run clang-tidy with the compiler." OFF)
if (NIGIRI_LINT)
  # clang-tidy will be run on all targets defined hereafter
  include(cmake/clang-tidy.cmake)
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(nigiri-compile-options
        -Weverything
        -Wno-c++98-compat
        -Wno-c++98-compat-pedantic
        -Wno-newline-eof
        -Wno-missing-prototypes
        -Wno-padded
        -Wno-double-promotion
        -Wno-undef
        -Wno-undefined-reinterpret-cast
        -Wno-float-conversion
        -Wno-global-constructors
        -Wno-exit-time-destructors
        -Wno-switch-enum
        -Wno-c99-designator
        -Wno-zero-as-null-pointer-constant
        -Wno-missing-noreturn
        -Wno-undefined-func-template
        -Wno-unsafe-buffer-usage
        -Wno-c++20-compat
        -Wno-reserved-macro-identifier
        -Wno-documentation-unknown-command
        -Wno-duplicate-enum
        -Wno-switch-default
        -Werror)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  set(nigiri-compile-options -Wall -Wextra -Werror)
elseif (MSVC)
  set(nigiri-compile-options /WX)
else ()
  set(nigiri-compile-options
        -Wall
        -Wextra
        -Wno-maybe-uninitialized)
  if (NOT CMAKE_CROSSCOMPILING)
    set(nigiri-compile-options ${nigiri-compile-options} -Werror)
  endif ()
endif ()


# --- MAIN ---
file(GLOB_RECURSE nigiri-server-files server/main.cc)
add_executable(nigiri-server ${nigiri-server-files})
target_link_libraries(nigiri-server PRIVATE nigiri)
target_compile_features(nigiri-server PUBLIC cxx_std_23)

# --- IMPORTER ---
file(GLOB_RECURSE nigiri-import-files exe/import.cc)
add_executable(nigiri-import ${nigiri-import-files})
target_link_libraries(nigiri-import PRIVATE nigiri boost-program_options ianatzdb-res)
target_compile_features(nigiri-import PUBLIC cxx_std_23)

# --- BENCHMARK ---
file(GLOB_RECURSE nigiri-benchmark-files exe/benchmark.cc)
add_executable(nigiri-benchmark ${nigiri-benchmark-files})
target_link_libraries(nigiri-benchmark PRIVATE nigiri boost-program_options ianatzdb-res)
target_compile_features(nigiri-benchmark PUBLIC cxx_std_23)

# --- QA ---
file(GLOB_RECURSE nigiri-qa-files exe/qa.cc)
add_executable(nigiri-qa ${nigiri-qa-files})
target_link_libraries(nigiri-qa PRIVATE nigiri boost-program_options)
target_compile_features(nigiri-qa PUBLIC cxx_std_23)

# --- TEST ---
add_library(nigiri-generated INTERFACE)
target_include_directories(nigiri-generated INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/generated)
configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/test/test_dir.h.in
      ${CMAKE_CURRENT_BINARY_DIR}/generated/test_dir.h
)
file(GLOB_RECURSE nigiri-test-files test/*.cc)
add_executable(nigiri-test ${nigiri-test-files})
target_include_directories(nigiri-test PRIVATE include)
target_link_libraries(nigiri-test PRIVATE nigiri gtest ianatzdb-res nigiri-generated)
target_compile_options(nigiri-test PRIVATE ${nigiri-compile-options})


# --- LIB ---
file(GLOB_RECURSE nigiri-files src/*.cc)
add_library(nigiri ${nigiri-files})
target_include_directories(nigiri PUBLIC include)
target_link_libraries(nigiri PUBLIC cista geo utl fmt date miniz date-tz wyhash unordered_dense gtfsrt oh opentelemetry_api pugixml)
target_compile_features(nigiri PUBLIC cxx_std_23)
target_compile_options(nigiri PRIVATE ${nigiri-compile-options})

if (NIGIRI_CUDA)
  if (MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -lcudadevrt")
  else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -lineinfo --compiler-options -static")
  endif()
  enable_language(CUDA)

  find_package(CUDAToolkit REQUIRED)
  file(GLOB_RECURSE nigiri-gpu-files src/*.cu)
  add_library(nigiri-gpu OBJECT ${nigiri-gpu-files})
  set_target_properties(nigiri-gpu PROPERTIES
      COMPILE_OPTIONS -G
      WINDOWS_EXPORT_ALL_SYMBOLS ON
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_ARCHITECTURES native
      CUDA_STANDARD 26
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
      INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/"
  )
  target_include_directories(nigiri-gpu PUBLIC include ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(nigiri-gpu PRIVATE cista fmt date unordered_dense geo utl)
  target_compile_definitions(nigiri-gpu PUBLIC NIGIRI_CUDA)

  target_link_libraries(nigiri-test PRIVATE nigiri-gpu)
  target_compile_definitions(nigiri PUBLIC NIGIRI_CUDA)
  set_property(TARGET nigiri-test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET nigiri PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()