#pragma once

#include <filesystem>

#include "geo/latlng.h"

#include "nigiri/types.h"

namespace nigiri {

class shape_test_mmap {
  using Key = shape_idx_t;
  using Value = geo::latlng;
  struct data_paths {
    explicit data_paths(std::string const& base_path)
        : data_file_{base_path + "-data.dat"},
          metadata_file_{base_path + "-metadata.dat"} {}
    ~data_paths() {
      for (auto const& path : {data_file_, metadata_file_}) {
        if (std::filesystem::exists(path)) {
          try {
            std::filesystem::remove(path);
          } catch (std::filesystem::filesystem_error const& e) {
            std::cerr << "Failed to delete '" << path << "': " << e.what()
                      << "\n";
          }
        }
      }
    }
    data_paths(data_paths const&) = delete;
    data_paths& operator=(data_paths const&) = delete;
    data_paths(data_paths&&) = delete;
    data_paths&& operator=(data_paths&&) = delete;
    std::string data_file_;
    std::string metadata_file_;
  };
  static shape_vecvec_t create_mmap(data_paths const& paths) {
    auto mode = cista::mmap::protection::WRITE;
    return {cista::basic_mmap_vec<Value, std::uint64_t>{
                cista::mmap{paths.data_file_.data(), mode}},
            cista::basic_mmap_vec<cista::base_t<Key>, std::uint64_t>{
                cista::mmap{paths.metadata_file_.data(), mode}}};
  }

public:
  explicit shape_test_mmap(std::string const& base_path)
      : paths_{base_path}, mmap_{create_mmap(paths_)} {}
  shape_vecvec_t& get_shape_data() { return mmap_; }

private:
  data_paths paths_;
  shape_vecvec_t mmap_;
};

}  // namespace nigiri