#include "nigiri/loader/dir.h"

#include <optional>
#include <variant>
#include <vector>

#include "miniz.h"

#include "cista/mmap.h"

#include "utl/parser/cstr.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "nigiri/logging.h"
#include "wyhash.h"

namespace nigiri::loader {

file::content::~content() = default;

dir::~dir() = default;
dir::dir(std::filesystem::path p) : path_{std::move(p)} {}
dir::dir(dir const&) = default;
dir::dir(dir&&) noexcept = default;
dir& dir::operator=(dir const&) = default;
dir& dir::operator=(dir&&) noexcept = default;

std::string normalize(std::filesystem::path const& p) {
  std::string s;
  auto first = true;
  for (auto const& el : p) {
    if (el == ".") {
      continue;
    }
    if (!first) {
      s += "/";
    }
    first = false;
    s += el.generic_string();
  }
  return s;
}

// --- File directory implementation ---
fs_dir::fs_dir(std::filesystem::path p) : dir{std::move(p)} {}
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
    mmap_content() = delete;
    mmap_content(mmap_content const&) = delete;
    mmap_content(mmap_content&&) = delete;
    mmap_content& operator=(mmap_content&&) = delete;
    mmap_content& operator=(mmap_content const&) = delete;
    explicit mmap_content(std::filesystem::path const& p)
        : mmap_{p.string().c_str(), cista::mmap::protection::READ} {
      log(log_lvl::info, "loader.fs_dir", "loaded {}: {} bytes", p,
          mmap_.size());
    }
    ~mmap_content() final = default;
    std::string_view get() const final { return mmap_.view(); }
    cista::mmap mmap_;
  };
  return file{(path_ / p).string(), std::make_unique<mmap_content>(path_ / p)};
}
bool fs_dir::exists(std::filesystem::path const& p) const {
  return std::filesystem::is_regular_file(path_ / p);
}
std::size_t fs_dir::file_size(std::filesystem::path const& p) const {
  return std::filesystem::file_size(path_ / p);
}
dir_type fs_dir::type() const { return dir_type::kFilesystem; }
std::uint64_t fs_dir::hash() const {
  auto h = std::uint64_t{0U};
  for (auto const& entry :
       std::filesystem::recursive_directory_iterator(path_)) {
    auto const data = cista::mmap{entry.path().generic_string().c_str(),
                                  cista::mmap::protection::READ};
    auto const name = entry.path().generic_string();
    h = wyhash(name.data(), name.size(), h, _wyp);
    h = wyhash(data.data(), data.size(), h, _wyp);
  }
  return h;
}

// --- ZIP directory implementation ---
std::optional<mz_uint32> find_file_idx(mz_zip_archive* ar,
                                       std::filesystem::path const& p) {
  mz_uint32 file_idx = 0U;
  auto const r = mz_zip_reader_locate_file_v2(ar, p.string().c_str(), nullptr,
                                              0, &file_idx);
  return r == MZ_TRUE ? std::make_optional<mz_uint32>(file_idx) : std::nullopt;
}
mz_uint32 get_file_idx(mz_zip_archive* ar, std::filesystem::path const& p) {
  auto const file_idx = find_file_idx(ar, p);
  utl::verify(file_idx.has_value(), "cannot locate file {} in zip", p);
  return *file_idx;
}
mz_zip_archive_file_stat get_stat(mz_zip_archive* ar,
                                  mz_uint32 const file_idx) {
  auto file_stat = mz_zip_archive_file_stat{};
  auto const r = mz_zip_reader_file_stat(ar, file_idx, &file_stat);
  utl::verify(r == MZ_TRUE, "zip: cannot stat file {}", file_idx);
  return file_stat;
}
std::filesystem::path get_file_path(mz_zip_archive* ar,
                                    mz_uint32 const file_idx) {
  return get_stat(ar, file_idx).m_filename;
}
struct zip_file_content final : public file::content {
  zip_file_content() = delete;
  zip_file_content(zip_file_content const&) = delete;
  zip_file_content(zip_file_content&&) = delete;
  zip_file_content& operator=(zip_file_content&&) = delete;
  zip_file_content& operator=(zip_file_content const&) = delete;
  ~zip_file_content() final;
  zip_file_content(mz_zip_archive* ar,
                   std::filesystem::path const& p,
                   mz_uint32 const file_idx) {
    buf_.resize(get_stat(ar, file_idx).m_uncomp_size);
    auto const r =
        mz_zip_reader_extract_to_mem(ar, file_idx, buf_.data(), buf_.size(), 0);
    utl::verify(r == MZ_TRUE, "cannot extract file {} from zip", p);
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
  impl() = delete;
  impl(impl const&) = delete;
  impl(impl&&) = delete;
  impl& operator=(impl&&) = delete;
  impl& operator=(impl const&) = delete;
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
    auto parent = p.string();
    if (!parent.ends_with('/') && !p.has_extension()) {
      parent += '/';
    }
    if (parent == "./" || parent == "/") {
      parent = "";
    } else {
      auto const file_idx = get_file_idx(&ar_, parent);
      auto const is_dir = mz_zip_reader_is_file_a_directory(&ar_, file_idx);
      if (is_dir != MZ_TRUE) {
        return {p};
      }
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
  bool exists(std::filesystem::path const& p) {
    return find_file_idx(&ar_, p).has_value();
  }
  std::size_t file_size(std::filesystem::path const& p) {
    auto const file_idx = find_file_idx(&ar_, p);
    utl::verify(file_idx.has_value(), "zip_dir::file_size: not found: {}", p);
    return get_stat(&ar_, *file_idx).m_uncomp_size;
  }

  mz_zip_archive ar_{};
  std::variant<std::vector<std::uint8_t>, cista::mmap> memory_;
};
zip_dir::zip_dir(std::filesystem::path const& p)
    : dir{p}, impl_{std::make_unique<impl>(p)} {}
zip_dir::zip_dir(std::vector<std::uint8_t> b)
    : dir{"::memory::"}, impl_{std::make_unique<impl>(std::move(b))} {}
zip_dir::~zip_dir() = default;
std::vector<std::filesystem::path> zip_dir::list_files(
    std::filesystem::path const& p) const {
  return impl_->list_files(normalize(p));
}
file zip_dir::get_file(std::filesystem::path const& p) const {
  return file{p.string(),
              std::make_unique<zip_file_content>(&impl_->ar_, normalize(p))};
}
bool zip_dir::exists(std::filesystem::path const& p) const {
  return impl_->exists(normalize(p));
}
std::size_t zip_dir::file_size(std::filesystem::path const& p) const {
  return impl_->file_size(normalize(p));
}
dir_type zip_dir::type() const { return dir_type::kZip; }
std::uint64_t zip_dir::hash() const {
  return std::visit(
      [](auto const& data) {
        return wyhash(data.data(), data.size(), 0, _wyp);
      },
      impl_->memory_);
}

// --- In-memory directory implementation ---
mem_dir::mem_dir(dir_t d) : dir{"::memory::"}, dir_{std::move(d)} {}
mem_dir::~mem_dir() = default;
mem_dir::mem_dir(mem_dir const&) = default;
mem_dir::mem_dir(mem_dir&&) noexcept = default;
mem_dir& mem_dir::operator=(mem_dir const&) = default;
mem_dir& mem_dir::operator=(mem_dir&&) noexcept = default;
mem_dir& mem_dir::add(std::pair<std::filesystem::path, std::string> f) {
  dir_.emplace(normalize(f.first), std::move(f.second));
  return *this;
}
std::vector<std::filesystem::path> mem_dir::list_files(
    std::filesystem::path const& dir) const {
  auto normal_dir = normalize(dir);
  if (normal_dir == ".") {
    normal_dir = "";
  }
  std::vector<std::filesystem::path> paths;
  for (auto const& [p, _] : dir_) {
    if (p.string().starts_with(normal_dir)) {
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
  return file{p.string(),
              std::make_unique<mem_file_content>(dir_.at(normalize(p)))};
}
bool mem_dir::exists(std::filesystem::path const& p) const {
  return dir_.contains(normalize(p));
}
std::size_t mem_dir::file_size(std::filesystem::path const& p) const {
  return dir_.at(normalize(p)).size();
}

std::unique_ptr<dir> make_dir(std::filesystem::path const& p) {
  if (std::filesystem::is_regular_file(p) && p.extension() == ".zip") {
    return std::make_unique<zip_dir>(p);
  } else if (std::filesystem::is_directory(p)) {
    return std::make_unique<fs_dir>(p);
  } else {
    throw utl::fail("path {} is neither a zip file nor a directory", p);
  }
}
dir_type mem_dir::type() const { return dir_type::kInMemory; }
std::uint64_t mem_dir::hash() const {
  auto h = std::uint64_t{0U};
  for (auto const& [path, data] : dir_) {
    auto const name = path.generic_string();
    h = wyhash(name.data(), name.size(), h, _wyp);
    h = wyhash(data.data(), data.size(), h, _wyp);
  }
  return h;
}

mem_dir mem_dir::read(std::string_view s) {
  std::string_view file_name;
  char const* file_content_begin = nullptr;
  auto dir = mem_dir::dir_t{};
  utl::for_each_line(s, [&](utl::cstr const line) {
    if (line.starts_with("#")) {
      if (file_content_begin != nullptr) {
        auto const length =
            static_cast<std::size_t>(line.begin() - file_content_begin - 1);
        dir.emplace(file_name, std::string{file_content_begin, length});
      }
      file_name = line.substr(1).trim();
      file_content_begin = line.end() + 1;
    }
  });
  if (file_content_begin != nullptr) {
    auto const length =
        static_cast<std::size_t>(s.data() + s.size() - file_content_begin);
    dir.emplace(file_name, std::string{file_content_begin, length});
  }
  return {dir};
}

}  // namespace nigiri::loader