#pragma once

#include <filesystem>

#include "nigiri/loader/gtfs/shape.h"

namespace nigiri::loader::gtfs {

inline shape::mmap_vecvec create_mmap_vecvec(std::vector<std::string>& paths) {
  auto mode = cista::mmap::protection::WRITE;
  return {cista::basic_mmap_vec<shape::stored_type, std::uint64_t>{
              cista::mmap{paths.at(0).data(), mode}},
          cista::basic_mmap_vec<cista::base_t<shape::key_type>, std::uint64_t>{
              cista::mmap{paths.at(1).data(), mode}}};
}

inline void cleanup_paths(std::vector<std::string> const& paths) {
  for (auto path : paths) {
    if (std::filesystem::exists(path)) {
      std::filesystem::remove(path);
    }
  }
}

inline auto create_temporary_paths(std::string base_path) {
  std::vector<std::string> paths{base_path + "-data.dat",
                                 base_path + "-metadata.dat"};
  return std::make_pair(create_mmap_vecvec(paths), paths);
}

}  // namespace nigiri::loader::gtfs