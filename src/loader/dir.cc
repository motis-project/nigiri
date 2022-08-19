#include "nigiri/loader/dir.h"

#include <variant>
#include <vector>

#include "miniz.h"

#include "cista/mmap.h"

#include "utl/to_vec.h"
#include "utl/verify.h"

namespace nigiri::loader {

file::content::~content() = default;

dir::~dir() = default;

// --- File directory implementation ---
fs_dir::fs_dir(std::filesystem::path p) : path_{std::move(p)} {}
fs_dir::~fs_dir() = default;
std::vector<std::filesystem::path> fs_dir::list_files(
    std::filesystem::path const& p) const {
  if (std::filesystem::is_directory(path_ / p)) {
    std::vector<std::filesystem::path> paths;
    for (auto const& e :
         std::filesystem::recursive_directory_iterator(path_ / p)) {
      paths.emplace_back(relative(e.path(), path_));
    }
    std::sort(begin(paths), end(paths));
    return paths;
  } else {
    return {p};
  }
}
file fs_dir::get_file(std::filesystem::path const& p) const {
  struct mmap_content final : public file::content {
    explicit mmap_content(std::filesystem::path const& p)
        : mmap_{p.string().c_str(), cista::mmap::protection::READ} {}
    ~mmap_content() final = default;
    std::string_view get() const final { return mmap_.view(); }
    cista::mmap mmap_;
  };
  return file{(path_ / p).string(), std::make_unique<mmap_content>(path_ / p)};
}

// --- ZIP directory implementation ---
mz_uint32 get_file_idx(mz_zip_archive* ar, std::filesystem::path const& p) {
  mz_uint32 file_idx;
  auto const r = mz_zip_reader_locate_file_v2(ar, p.string().c_str(), nullptr,
                                              0, &file_idx);
  utl::verify(r, "cannot locate file {} in zip", p);
  return file_idx;
}
std::filesystem::path get_file_path(mz_zip_archive* ar,
                                    mz_uint32 const file_idx) {
  auto file_stat = mz_zip_archive_file_stat{};
  auto const r = mz_zip_reader_file_stat(ar, file_idx, &file_stat);
  utl::verify(r, "zip: cannot determine file size of {}", file_idx);
  return file_stat.m_filename;
}
struct zip_file_content final : public file::content {
  ~zip_file_content() final;
  zip_file_content(mz_zip_archive* ar,
                   std::filesystem::path const& p,
                   mz_uint32 const file_idx) {
    mz_bool r;

    mz_zip_archive_file_stat file_stat{};
    r = mz_zip_reader_file_stat(ar, file_idx, &file_stat);
    utl::verify(r, "cannot read file size of {} in zip", p);

    buf_.resize(file_stat.m_uncomp_size);
    r = mz_zip_reader_extract_to_mem(ar, file_idx, buf_.data(), buf_.size(), 0);
    utl::verify(r, "cannot extract file {} from zip", p);
  }
  zip_file_content(mz_zip_archive* ar, std::filesystem::path const& p)
      : zip_file_content{ar, p, get_file_idx(ar, p)} {}
  std::string_view get() const final {
    return {reinterpret_cast<char const*>(buf_.data()), buf_.size()};
  }
  std::vector<std::uint8_t> buf_;
};
zip_file_content::~zip_file_content() = default;
struct zip_dir::impl {
  explicit impl(std::vector<std::uint8_t> b) : memory_{std::move(b)} {
    open("inmemory");
  }
  explicit impl(std::filesystem::path const& p)
      : memory_{
            cista::mmap{p.string().c_str(), cista::mmap::protection::READ}} {
    open(p);
  }
  ~impl() { mz_zip_reader_end(&ar_); }
  void open(std::filesystem::path const& p) {
    std::visit(
        [&](auto&& buf) {
          mz_zip_zero_struct(&ar_);
          auto r = mz_zip_reader_init_mem(&ar_, buf.data(), buf.size(), 0);
          utl::verify(r, "unable to open zip at {}, error={}", p, r);
        },
        memory_);
  }

  std::vector<std::filesystem::path> list_files(
      std::filesystem::path const& p) {
    auto const parent = p.string();

    auto const file_idx = get_file_idx(&ar_, p);
    auto const is_dir = mz_zip_reader_is_file_a_directory(&ar_, file_idx);
    if (!is_dir) {
      return {p};
    }

    std::vector<std::filesystem::path> files;
    auto const num_files = mz_zip_reader_get_num_files(&ar_);
    for (auto i = mz_uint32{0U}; i != num_files; ++i) {
      auto path = get_file_path(&ar_, i);
      auto const p_str = path.string();
      if (p_str.starts_with(parent) && p_str != parent) {
        files.emplace_back(std::move(path));
      }
    }
    std::sort(begin(files), end(files));
    return files;
  }

  mz_zip_archive ar_;
  std::variant<std::vector<std::uint8_t>, cista::mmap> memory_;
};
zip_dir::zip_dir(std::filesystem::path const& p)
    : impl_{std::make_unique<impl>(p)} {}
zip_dir::zip_dir(std::vector<std::uint8_t> b)
    : impl_{std::make_unique<impl>(std::move(b))} {}
zip_dir::~zip_dir() = default;
std::vector<std::filesystem::path> zip_dir::list_files(
    std::filesystem::path const& p) const {
  return impl_->list_files(p);
}
file zip_dir::get_file(std::filesystem::path const& p) const {
  return file{p.string(), std::make_unique<zip_file_content>(&impl_->ar_, p)};
}

// --- In-memory directory implementation ---
mem_dir::mem_dir(dir_t d) : dir_{std::move(d)} {}
mem_dir::~mem_dir() = default;
std::vector<std::filesystem::path> mem_dir::list_files(
    std::filesystem::path const& dir) const {
  std::vector<std::filesystem::path> paths;
  for (auto const& [p, _] : dir_) {
    if (p.string().starts_with(dir.string())) {
      paths.emplace_back(p);
    }
  }
  return paths;
}
file mem_dir::get_file(std::filesystem::path const& p) const {
  struct mem_file_content : public file::content {
    explicit mem_file_content(std::string const& b) : buf_{b} {}
    std::string_view get() const final { return buf_; }
    std::string const& buf_;
  };
  return file{p.string(), std::make_unique<mem_file_content>(dir_.at(p))};
}

}  // namespace nigiri::loader