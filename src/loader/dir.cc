#include "nigiri/loader/dir.h"

#include "cista/mmap.h"

namespace nigiri::loader {

struct mmap_content final : public file::content {
  explicit mmap_content(std::string const& path) : mmap_{path.c_str()} {}
  ~mmap_content() final = default;
  std::string_view get() final {
    return {reinterpret_cast<char const*>(mmap_.data()), mmap_.size()};
  }
  cista::mmap mmap_;
};

fs_dir::~fs_dir() = default;

file fs_dir::get_file(std::string_view path) {}

zip_dir::~zip_dir() = default;

file zip_dir::get_file(std::string_view path) {}

}  // namespace nigiri::loader