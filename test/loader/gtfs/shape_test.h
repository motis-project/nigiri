#pragma once

#include <filesystem>

#include "geo/latlng.h"

#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

class shape_test_mmap {
  using Key = shape_idx_t;
  using Value = geo::latlng;
  static auto create_paths(std::string base_path) {
    return std::vector<std::string>{base_path + "-data.dat",
                                    base_path + "-metadata.dat"};
  }
  static mm_vecvec<Key, Value> create_mmap(std::vector<std::string>& paths) {
    auto mode = cista::mmap::protection::WRITE;
    return {cista::basic_mmap_vec<Value, std::uint64_t>{
                cista::mmap{paths.at(0).data(), mode}},
            cista::basic_mmap_vec<cista::base_t<Key>, std::uint64_t>{
                cista::mmap{paths.at(1).data(), mode}}};
  }

public:
  explicit shape_test_mmap(std::string base_path)
      : paths{create_paths(base_path)}, mmap{create_mmap(paths)} {}
  ~shape_test_mmap() {
    for (auto path : paths) {
      if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
      }
    }
  }
  mm_vecvec<Key, Value>& get_vecvec() { return mmap; }

private:
  std::vector<std::string> paths;
  mm_vecvec<Key, Value> mmap;
};

}  // namespace nigiri::loader::gtfs