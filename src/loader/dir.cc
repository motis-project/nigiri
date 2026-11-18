#include "nigiri/loader/dir.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <ranges>
#include <vector>

#include "fmt/ostream.h"

#include "mz.h"
#include "mz_strm.h"
#include "mz_zip.h"
#include "mz_zip_rw.h"

#include "cista/mmap.h"

#include "utl/parser/cstr.h"
#include "utl/verify.h"

#include "wyhash.h"

template <>
struct fmt::formatter<std::filesystem::path> : ostream_formatter {};

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
        : mmap_{p.string().c_str(), cista::mmap::protection::READ} {}
    ~mmap_content() final = default;
    std::string_view get() const final { return mmap_.view(); }
    void* get_mutable() override { throw utl::fail("not mutable"); }
    bool is_mutable() const override { return false; }
    std::size_t size() const override { return mmap_.size(); }
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

// Read-only mz_stream over a byte vector. Same as mz_stream_mem but with
// 64-bit positions. mz_zip_reader_open_buffer only supports 32-bit positions.
struct mz_stream_vec {
  mz_stream base{};
  std::vector<std::uint8_t> const* buf{nullptr};
  std::int64_t pos{0};
};

static void* mz_stream_vec_create();
static void mz_stream_vec_destroy(void** stream) {
  free(*stream);
  *stream = nullptr;
}
static std::int32_t mz_stream_vec_open(void*, char const*, std::int32_t) {
  return MZ_OK;
}
static std::int32_t mz_stream_vec_is_open(void*) { return MZ_OK; }
static std::int32_t mz_stream_vec_read(void* stream, void* b,
                                       std::int32_t size) {
  auto* s = static_cast<mz_stream_vec*>(stream);
  auto const remaining = static_cast<std::int64_t>(s->buf->size()) - s->pos;
  auto const n = static_cast<std::int32_t>(
      std::min<std::int64_t>(size, std::max<std::int64_t>(remaining, 0)));
  if (n <= 0) {
    return 0;
  }
  std::memcpy(b, s->buf->data() + s->pos, static_cast<std::size_t>(n));
  s->pos += n;
  return n;
}
static std::int32_t mz_stream_vec_write(void*, void const*, std::int32_t) {
  return MZ_READ_ERROR;
}
static std::int64_t mz_stream_vec_tell(void* stream) {
  return static_cast<mz_stream_vec*>(stream)->pos;
}
static std::int32_t mz_stream_vec_seek(void* stream, std::int64_t offset,
                                       std::int32_t origin) {
  auto* s = static_cast<mz_stream_vec*>(stream);
  auto new_pos = s->pos;
  switch (origin) {
    case MZ_SEEK_CUR: {
      new_pos += offset;
      break;
    }
    case MZ_SEEK_END: {
      new_pos = static_cast<std::int64_t>(s->buf->size()) + offset;
      break;
    }
    case MZ_SEEK_SET: {
      new_pos = offset;
      break;
    }
    default: {
      return MZ_SEEK_ERROR;
    }
  }
  if (new_pos < 0 || new_pos > static_cast<std::int64_t>(s->buf->size())) {
    return MZ_SEEK_ERROR;
  }
  s->pos = new_pos;
  return MZ_OK;
}
static std::int32_t mz_stream_vec_close(void*) { return MZ_OK; }
static std::int32_t mz_stream_vec_error(void*) { return MZ_OK; }

static mz_stream_vtbl mz_stream_vec_vtbl = {
    mz_stream_vec_open,    mz_stream_vec_is_open, mz_stream_vec_read,
    mz_stream_vec_write,   mz_stream_vec_tell,    mz_stream_vec_seek,
    mz_stream_vec_close,   mz_stream_vec_error,   mz_stream_vec_create,
    mz_stream_vec_destroy, nullptr,               nullptr};

static void* mz_stream_vec_create() {
  auto* s = static_cast<mz_stream_vec*>(calloc(1, sizeof(mz_stream_vec)));
  if (s != nullptr) {
    s->base.vtbl = &mz_stream_vec_vtbl;
  }
  return s;
}

std::string mz_code_str(int32_t const code) {
  switch (code) {
    case MZ_OK: {
      return "ok";
    }
    case MZ_FORMAT_ERROR: {
      return "format error";
    }
    case MZ_INTERNAL_ERROR: {
      return "internal error";
    }
    case MZ_OPEN_ERROR: {
      return "open error";
    }
    case MZ_READ_ERROR: {
      return "read error";
    }
    case MZ_SEEK_ERROR: {
      return "seek error";
    }
    case MZ_TELL_ERROR: {
      return "tell error";
    }
    case MZ_CRC_ERROR: {
      return "crc error";
    }
    case MZ_MEM_ERROR: {
      return "memory error";
    }
    case MZ_SUPPORT_ERROR: {
      return "unsupported compression method";
    }
    case MZ_DATA_ERROR: {
      return "data error";
    }
    case MZ_END_OF_LIST: {
      return "end of list";
    }
    case MZ_EXIST_ERROR: {
      return "entry does not exist";
    }
    case MZ_PASSWORD_ERROR: {
      return "password required";
    }
    case MZ_SIGN_ERROR: {
      return "signature error";
    }
    case MZ_SYMLINK_ERROR: {
      return "symlink error";
    }
    case MZ_PARAM_ERROR: {
      return "parameter error";
    }
    case MZ_CRYPT_ERROR: {
      return "crypt error";
    }
    default: {
      return fmt::to_string(code);
    }
  }
}
mz_zip_file* entry_info(void* reader, std::string const& p) {
  auto* info = static_cast<mz_zip_file*>(nullptr);
  auto const err = mz_zip_reader_entry_get_info(reader, &info);
  utl::verify(err == MZ_OK && info != nullptr, "zip: cannot get info for {}: {}",
              p, mz_code_str(err));
  return info;
}

mz_zip_file* locate(void* reader, std::string const& p) {
  auto const err = mz_zip_reader_locate_entry(reader, p.c_str(), 0);
  if (err == MZ_END_OF_LIST) {
    return nullptr;
  }
  utl::verify(err == MZ_OK, "zip: cannot locate file {}: {}", p, mz_code_str(err));
  return entry_info(reader, p);
}
struct zip_file_content final : public file::content {
  zip_file_content() = delete;
  zip_file_content(zip_file_content const&) = delete;
  zip_file_content(zip_file_content&&) = delete;
  zip_file_content& operator=(zip_file_content&&) = delete;
  zip_file_content& operator=(zip_file_content const&) = delete;
  ~zip_file_content() final = default;
  zip_file_content(void* reader, std::filesystem::path const& p) {
    auto const info = locate(reader, p.string());
    utl::verify(info != nullptr, "zip: cannot locate file {} in zip", p);
    auto const size = static_cast<std::size_t>(info->uncompressed_size);

    buf_.resize(size);
    auto err = mz_zip_reader_entry_open(reader);
    utl::verify(err == MZ_OK, "zip: cannot open file {}: {}", p, mz_code_str(err));

    constexpr auto kMaxChunk =
        static_cast<std::size_t>(std::numeric_limits<int32_t>::max());
    auto offset = std::size_t{0U};
    while (offset < size) {
      auto const chunk =
          static_cast<int32_t>(std::min(size - offset, kMaxChunk));
      auto const n =
          mz_zip_reader_entry_read(reader, buf_.data() + offset, chunk);
      utl::verify(n >= 0, "zip: cannot extract file {}: {}", p, mz_code_str(n));
      if (n == 0) {
        break;  // premature end of entry
      }
      offset += static_cast<std::size_t>(n);
    }
    utl::verify(offset == size,
                "zip: unexpected size for {}: read {} of {} bytes", p, offset,
                size);

    err = mz_zip_reader_entry_close(reader);
    utl::verify(err == MZ_OK, "zip: cannot close file {}: {}", p, mz_code_str(err));
  }
  std::string_view get() const final {
    return {reinterpret_cast<char const*>(buf_.data()), buf_.size()};
  }
  void* get_mutable() override { return reinterpret_cast<void*>(buf_.data()); }
  bool is_mutable() const override { return true; }
  std::size_t size() const override { return buf_.size(); }
  std::vector<std::uint8_t> buf_;
};
struct zip_dir::impl {
  impl() = delete;
  impl(impl const&) = delete;
  impl(impl&&) = delete;
  impl& operator=(impl&&) = delete;
  impl& operator=(impl const&) = delete;
  explicit impl(std::filesystem::path const& p)
      : reader_{mz_zip_reader_create()} {
    utl::verify(reader_ != nullptr, "zip: cannot create reader for {}", p);
    auto const err = mz_zip_reader_open_file(reader_, p.string().c_str());
    utl::verify(err == MZ_OK, "unable to open zip at {}, error={}", p,
                mz_code_str(err));
  }
  explicit impl(std::vector<std::uint8_t> b)
      : memory_{std::move(b)},
        reader_{mz_zip_reader_create()},
        stream_{mz_stream_vec_create()},
        from_memory_{true} {
    utl::verify(reader_ != nullptr,
                "zip: cannot create reader for in-memory archive");
    utl::verify(stream_ != nullptr,
                "zip: cannot create stream for in-memory archive");
    static_cast<mz_stream_vec*>(stream_)->buf = &memory_;
    auto const err = mz_zip_reader_open(reader_, stream_);
    utl::verify(err == MZ_OK, "unable to open in-memory zip, error={}",
                mz_code_str(err));
  }
  ~impl() {
    if (reader_ != nullptr) {
      mz_zip_reader_delete(&reader_);
    }
    if (stream_ != nullptr) {
      mz_stream_delete(&stream_);
    }
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
      utl::verify(locate(reader_, parent) != nullptr,
                  "zip: cannot locate file {} in zip", parent);
      if (mz_zip_reader_entry_is_dir(reader_) != MZ_OK) {
        return {p};
      }
    }

    std::vector<std::filesystem::path> files;
    auto err = mz_zip_reader_goto_first_entry(reader_);
    while (err == MZ_OK) {
      auto const name = std::string{entry_info(reader_, parent)->filename};
      if (name.starts_with(parent) && name != parent) {
        files.emplace_back(name);
      }
      err = mz_zip_reader_goto_next_entry(reader_);
    }
    utl::verify(err == MZ_END_OF_LIST, "zip: error while listing {}: {}", p,
                mz_code_str(err));
    std::sort(begin(files), end(files));
    return files;
  }
  bool exists(std::filesystem::path const& p) {
    return locate(reader_, p.string()) != nullptr;
  }
  std::size_t file_size(std::filesystem::path const& p) {
    auto const info = locate(reader_, p.string());
    utl::verify(info != nullptr, "zip_dir::file_size: not found: {}", p);
    return static_cast<std::size_t>(info->uncompressed_size);
  }

  std::vector<std::uint8_t> memory_;
  void* reader_{nullptr};
  void* stream_{nullptr};
  bool from_memory_{false};
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
              std::make_unique<zip_file_content>(impl_->reader_, normalize(p))};
}
bool zip_dir::exists(std::filesystem::path const& p) const {
  return impl_->exists(normalize(p));
}
std::size_t zip_dir::file_size(std::filesystem::path const& p) const {
  return impl_->file_size(normalize(p));
}
dir_type zip_dir::type() const { return dir_type::kZip; }
std::uint64_t zip_dir::hash() const {
  if (impl_->from_memory_) {
    return wyhash(impl_->memory_.data(), impl_->memory_.size(), 0U, _wyp);
  }
  auto const data =
      cista::mmap{path_.string().c_str(), cista::mmap::protection::READ};
  return wyhash(data.data(), data.size(), 0U, _wyp);
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
    void* get_mutable() override { throw utl::fail("not mutable"); }
    bool is_mutable() const override { return false; }
    std::size_t size() const override { return buf_.size(); }
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
  auto ext = p.extension().string();
  std::ranges::transform(ext, ext.begin(), ::tolower);
  if (std::filesystem::is_regular_file(p) && ext == ".zip") {
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
